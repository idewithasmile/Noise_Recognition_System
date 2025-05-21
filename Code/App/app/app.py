import os
import logging
import asyncio
import signal
import sys
import threading
import platform
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Import các module
from core import Config
from services import FlaskService, YamnetService, ASTService, SSEService
from storage.audio_recorder import AudioRecorder
from websocket_server import WebSocketServer
from websocket_server import handlers
from api.routes.audio_routes import audio_bp, ensure_processing_thread

class Application:
    def __init__(self):
        self.config = Config()
        self.logger = self.config.logger
        self.loop = None
        self.executor = None
        self.flask_thread = None
        self.websocket_server = None
        self.sse_service = SSEService()
        self.shutdown_event = threading.Event()
        self.thread_local = threading.local()
        
        # Đảm bảo thư mục logs tồn tại
        self._ensure_directories()

    def _ensure_directories(self):
        """Đảm bảo các thư mục cần thiết tồn tại"""
        # Tạo các thư mục mặc định nếu không có trong config
        base_dir = Path(__file__).parent.parent
        directories = [
            base_dir / "logs",
            base_dir / "uploads",
            base_dir / "temp"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_event_loop(self):
        """Lấy event loop hiện tại hoặc tạo mới cho thread hiện tại"""
        if not hasattr(self.thread_local, 'loop') or self.thread_local.loop is None:
            if platform.system() == 'Windows':
                # Windows cần policy mới cho event loop
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            self.thread_local.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.thread_local.loop)
        return self.thread_local.loop

    def run_flask_in_thread(self, flask_service):
        """Chạy Flask trong thread riêng với context phù hợp"""
        def flask_thread_func():
            try:
                # Đảm bảo tạo event loop cho thread Flask
                loop = self.get_event_loop()
                flask_service.run(debug=False, use_reloader=False, threaded=True)
            except Exception as e:
                self.logger.error(f"Lỗi trong Flask thread: {e}")
                self.shutdown()

        self.flask_thread = threading.Thread(
            target=flask_thread_func,
            daemon=True
        )
        self.flask_thread.start()
        return self.flask_thread

    async def run_websocket_server(self):
        """Chạy WebSocket server trong coroutine riêng"""
        try:
            await self.websocket_server.run_forever()
        except asyncio.CancelledError:
            self.logger.info("Nhận yêu cầu dừng dịch vụ WebSocket")
        except Exception as e:
            self.logger.error(f"Lỗi trong dịch vụ WebSocket: {e}")
            raise

    def start_async_services(self, flask_app):
        """Khởi chạy các dịch vụ async trong event loop riêng"""
        # Tạo event loop cho thread chính
        self.loop = self.get_event_loop()
        
        # Khởi tạo WebSocket server với context của Flask
        self.websocket_server = WebSocketServer(self.config).set_handler(handlers.handle_client)
        
        # Chia sẻ event loop với handlers module
        # handlers.set_event_loop(self.loop)
        
        # Chạy các dịch vụ async trong thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        def run_async_loop():
            """Chạy event loop trong một thread riêng"""
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self.run_websocket_server())
            except Exception as e:
                self.logger.error(f"Lỗi trong async thread: {e}")
                self.shutdown()
        
        # Chạy event loop trong thread riêng
        self.executor.submit(run_async_loop)
        
        # Đảm bảo thread xử lý audio queue được khởi tạo
        with flask_app.app_context():
            ensure_processing_thread()

    def shutdown(self):
        """Dừng ứng dụng một cách an toàn"""
        if self.shutdown_event.is_set():
            return  # Tránh gọi shutdown nhiều lần
            
        self.logger.info("Bắt đầu quá trình dừng ứng dụng...")
        self.shutdown_event.set()

        # Đóng tất cả kết nối SSE
        self.sse_service.close_all_connections()

        # Dừng WebSocket server
        if self.websocket_server and self.loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket_server.stop(), 
                    self.loop
                )
                future.result(timeout=5)  # Đợi tối đa 5 giây
            except Exception as e:
                self.logger.error(f"Lỗi khi dừng WebSocket server: {e}")

        # Dừng event loop
        if self.loop:
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
                # Đợi loop dừng hoàn toàn
                if threading.current_thread() is threading.main_thread():
                    if self.loop.is_running():
                        self.logger.warning("Event loop vẫn đang chạy khi đóng")
                self.loop.close()
            except Exception as e:
                self.logger.error(f"Lỗi khi dừng event loop: {e}")

        # Dừng thread pool
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

        self.logger.info("Ứng dụng đã dừng.")

    def setup_signal_handlers(self):
        """Thiết lập các bộ xử lý tín hiệu"""
        def signal_handler(sig, frame):
            self.logger.info(f"Nhận tín hiệu {signal.Signals(sig).name}...")
            self.shutdown()
            sys.exit(0)

        # Chỉ thiết lập signal handlers trên Unix-like systems
        if platform.system() != 'Windows':
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    def main(self):
        """Hàm chính khởi chạy ứng dụng"""
        try:
            self.logger.info("Khởi động ứng dụng...")
            
            # Khởi tạo các service
            yamnet_service = YamnetService(self.config).load_model()
            ast_service = ASTService(self.config).load_model()
            recorder = AudioRecorder(self.config)

            flask_service = FlaskService(self.config).setup(
                yamnet_service=yamnet_service,
                ast_service=ast_service,
                recorder=recorder,
                sse_service=self.sse_service
            )
            
            # Đăng ký blueprint audio_bp vào Flask service
            flask_app = flask_service.get_app()
            # flask_app.register_blueprint(audio_bp, url_prefix='/api/audio')

            # Đăng ký SSE service vào Flask app
            self.sse_service.init_app(flask_app)

            # Thiết lập bộ xử lý tín hiệu
            self.setup_signal_handlers()

            # Khởi chạy các dịch vụ
            self.start_async_services(flask_app)
            self.run_flask_in_thread(flask_service)

            self.logger.info("Ứng dụng đã khởi động thành công!")

            # Chờ tín hiệu dừng
            while not self.shutdown_event.wait(1):
                pass

        except Exception as e:
            self.logger.error(f"Lỗi khởi chạy ứng dụng: {e}")
            self.shutdown()
            sys.exit(1)

if __name__ == "__main__":
    app = Application()
    app.main()