import os
import torch
from torch.utils.data import DataLoader
from ast_service import ASTService, ModelConfig

# Đảm bảo rằng đường dẫn đến mô hình và tệp nhãn là chính xác
MODEL_PATH = "D:\\NCKH\\server\\models\\ast\\ast_fsd50k_i2s_model.pt"
LABELS_FILE = "D:\\NCKH\\server\\models\\ast\\fsd50k_labels.json"

def test_predict_audio():
    
    # Khởi tạo cấu hình và dịch vụ AST
    config = ModelConfig()  # Tạo đối tượng cấu hình
    ast_service = ASTService(config)  # Tạo dịch vụ AST
    
    try:
        # Tải mô hình và lưu trữ vào thuộc tính của đối tượng
        ast_service = ASTService(config).load_model()
        
        if ast_service.model is None:
            print("Không thể tải mô hình.")
            return
        
        print(f"Đã tải mô hình và nhãn, có {len(ast_service.idx_to_label)} nhãn.")

        # Chọn tệp âm thanh để kiểm thử
        audio_file = r"D:\NCKH\server\testing\sample_audio\trumpet.wav"
        
        # Kiểm tra dự đoán
        predictions = ast_service.predict(audio_file)
        
        if not predictions:
            print("Không có dự đoán nào.")
        else:
            print("Dự đoán từ mô hình:")
            for label, confidence in predictions:
                print(f"{label}: {confidence:.2f}")
    
    except Exception as e:
        print(f"Lỗi khi chạy: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predict_audio()