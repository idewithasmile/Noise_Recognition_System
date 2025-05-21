from flask import Flask, render_template, send_from_directory
import os

class FlaskService:
    """Module quản lý Flask service"""
    
    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('flask')

        # Xác định đường dẫn template
        server_dir = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục của file hiện tại
        template_path = os.path.join(os.path.dirname(server_dir), 'templates')  # Lên một cấp và vào templates

        self.app = Flask(__name__, template_folder=template_path)
        self.logger.info("Đã khởi tạo Flask service")
        
    def setup(self, yamnet_service=None, ast_service=None, recorder =None, sse_service=None):
        """Thiết lập các route và middleware cho Flask"""

        from api.routes.audio_routes import audio_bp

        # Đăng ký các service vào app config để có thể sử dụng trong các route
        self.app.config['yamnet_service'] = yamnet_service
        self.app.config['ast_service'] = ast_service
        self.app.config['recorder'] = recorder
        self.app.config['sse_service'] = sse_service
        self.app.config['DELETE_AFTER_PROCESS'] = self.config.DELETE_AFTER_PROCESS
    
        self.app.register_blueprint(audio_bp, url_prefix='/api/audio')
        
        @self.app.route('/')
        def index():
            # return {"status": "online", "message": "Hệ thống đang hoạt động"}
    
            return render_template('index.html')
            
        
        @self.app.route('/api/status')
        def status():
            return {
                "status": "online", 
                "websocket": f"ws://{self.config.HOST}:{self.config.WEBSOCKET_PORT}"
            }
        
        self.logger.info("Đã thiết lập các route cho Flask service")
        return self
        
    def run(self, debug=False, use_reloader=False, threaded=True):
        """Chạy Flask server"""
        self.logger.info(f"Khởi động Flask server tại http://{self.config.HOST}:{self.config.FLASK_PORT}")
        self.app.run(
            host=self.config.HOST, 
            port=self.config.FLASK_PORT, 
            debug=debug,
            use_reloader=use_reloader,
            threaded=threaded
        )
    
    def get_app(self):
        """Trả về đối tượng Flask app"""
        return self.app