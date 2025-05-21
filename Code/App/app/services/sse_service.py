import json
import queue
from flask import Response
from threading import Lock
from core.utils import numpy_to_builtin_type  # Đảm bảo bạn có hàm này để xử lý numpy types

class SSEService:
    def __init__(self):
        self.clients = []
        self.lock = Lock()

    def init_app(self, app):
        @app.route('/api/events')
        def sse_stream():
            def generate():
                client_queue = queue.Queue()

                with self.lock:
                    self.clients.append(client_queue)

                try:
                    # Gửi sự kiện xác nhận kết nối ban đầu
                    yield 'data: {"message": "connected"}\n\n'

                    while True:
                        message = client_queue.get()
                        if message is None:
                            break
                        clean_data = numpy_to_builtin_type(message)
                        yield f'data: {json.dumps(clean_data)}\n\n'
                except GeneratorExit:
                    pass
                finally:
                    with self.lock:
                        if client_queue in self.clients:
                            self.clients.remove(client_queue)

            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )

    def send_event(self, data):
        """Gửi dữ liệu đến tất cả client đang kết nối."""
        with self.lock:
            clients_copy = self.clients.copy()

        for client_queue in clients_copy:
            try:
                client_queue.put_nowait(data)
            except Exception as e:
                print(f"[ERROR] Gửi message thất bại: {e}")

    def close_all_connections(self):
        """Đóng tất cả kết nối SSE."""
        with self.lock:
            for client_queue in self.clients:
                try:
                    client_queue.put_nowait(None)
                except Exception as e:
                    print(f"[ERROR] Đóng kết nối thất bại: {e}")
            self.clients.clear()
