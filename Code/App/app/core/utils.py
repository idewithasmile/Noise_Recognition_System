# Các tiện ích
import os
from datetime import datetime
import numpy as np
import json

def extract_timestamp_from_filename(filename):
    """Trích xuất timestamp từ tên file"""
    try:
        # Lấy phần tên file (bỏ đường dẫn)
        base_name = os.path.basename(filename)
        
        # Tách phần timestamp từ tên file
        # Ví dụ: "audio_20230515_123456_789000.wav"
        parts = base_name.split('_')
        if len(parts) >= 3:
            date_part = parts[1]  # "20230515"
            time_part = parts[2]  # "123456"
            microseconds_part = parts[3].split('.')[0]  # "789000"

            # Kết hợp thành chuỗi datetime
            timestamp_str = f"{date_part} {time_part}.{microseconds_part}"
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d %H%M%S.%f")
            return timestamp.isoformat()  # Trả về dạng ISO 8601
    except Exception as e:
        print(f"[!] Lỗi khi trích xuất timestamp từ file {filename}: {e}")
        return None

def numpy_to_builtin_type(obj):

    if isinstance(obj, dict):
        return {k: numpy_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_builtin_type(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_builtin_type(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    # Nếu là kiểu cơ bản Python, trả về nguyên bản
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Nếu kiểu không biết, raise lỗi (có thể sửa hoặc log nếu muốn)
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


def normalize_predictions(preds):
    # preds là list tuple (label, score), chuyển thành list dict với float chuẩn
    return [{"label": label, "score": float(score)} for label, score in preds]


