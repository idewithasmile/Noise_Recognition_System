import os
import logging

class Config:
    """Cấu hình chung cho toàn hệ thống"""
    
    def __init__(self):
        # Thiết lập logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('server')
        
        # Cấu hình chung
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.FLASK_PORT = int(os.getenv('FLASK_PORT', 8000))
        self.WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', 5000))
        self.DELETE_AFTER_PROCESS = os.getenv('DELETE_AFTER_PROCESS', 'true').lower() == 'true'
        
        # Thư mục lưu trữ
        self.STORAGE_DIR = os.getenv('STORAGE_DIR', 'uploads')
        self.AUDIO_DIR = os.path.join(self.STORAGE_DIR, 'audio')
        
        # Đảm bảo các thư mục tồn tại
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        
        self.logger.info("Đã khởi tạo cấu hình hệ thống")

    def get_logger(self, name):
        """Tạo logger cho module cụ thể"""
        return logging.getLogger(f'server.{name}')