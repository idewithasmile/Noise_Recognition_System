import asyncio
import websockets

class WebSocketServer:
    """Module quản lý WebSocket server"""

    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('websocket')
        self.server = None
        self.handler = None
        self.logger.info("Đã khởi tạo WebSocket service")

    def set_handler(self, handler):
        """Thiết lập hàm xử lý kết nối"""
        self.handler = handler
        return self

    async def start(self):
        """Khởi động WebSocket server"""
        if not self.handler:
            self.logger.error("Chưa thiết lập handler cho WebSocket!")
            return

        self.logger.info(f"Khởi động WebSocket server tại ws://{self.config.HOST}:{self.config.WEBSOCKET_PORT}")
        self.server = await websockets.serve(
            self.handler,
            self.config.HOST,
            self.config.WEBSOCKET_PORT
        )
        self.logger.info("WebSocket server đang chạy")

    async def run_forever(self):
        """Chạy WebSocket server vô thời hạn"""
        await self.start()
        self._forever_future = asyncio.Future()
        await self._forever_future  # Giữ server chạy mãi

    async def stop(self):
        """Dừng WebSocket server một cách an toàn"""
        if self.server:
            self.logger.info("Đang dừng WebSocket server...")
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server đã dừng.")
        # Đảm bảo future được hủy để run_forever kết thúc
        if hasattr(self, '_forever_future') and not self._forever_future.done():
            self._forever_future.cancel()
