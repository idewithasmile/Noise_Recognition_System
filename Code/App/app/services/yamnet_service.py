import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import os

class YamnetService:
    """Module xử lý phân loại âm thanh với YAMNet"""
    
    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('yamnet')
        self.model = None
        self.labels = []
        self.model_path = os.path.join(os.path.dirname(__file__), "../models/yamnet")
        self.labels_path = os.path.join(os.path.dirname(__file__), "../models/yamnet/yamnet_class_map.csv")
        self.logger.info("Đã khởi tạo YAMNet service")
        
    def load_model(self):
        """Tải model YAMNet và nhãn phân loại"""
        self.logger.info("Đang tải YAMNet model...")
        
        try:
            # Load YAMNet model
            self.model = tf.saved_model.load(self.model_path)
            
            # Load labels nếu có
            if os.path.exists(self.labels_path):
                self.labels = pd.read_csv(self.labels_path)["display_name"].tolist()
                self.logger.info(f"Đã tải {len(self.labels)} nhãn phân loại")
            
            self.logger.info("Đã tải xong YAMNet model")
            return self
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải YAMNet model: {str(e)}")
            raise

    def _pad_waveform(self, waveform, target_length=16000):
        """Padding để đảm bảo chiều dài hợp lệ"""
        num_samples = len(waveform)
        if num_samples < target_length:
            waveform = np.concatenate([waveform, np.zeros(target_length - num_samples)])
        return waveform

    def predict(self, audio_data, sample_rate=16000, threshold=0.8, top_n=3):
        """
        Phân loại dữ liệu âm thanh
        
        Args:
            audio_data: Dữ liệu âm thanh dạng numpy array hoặc file path
            sample_rate: Tần số lấy mẫu (mặc định 16000)
            threshold: Ngưỡng tin cậy (0-1)
            top_n: Số kết quả trả về nếu không có kết quả nào đạt ngưỡng
            
        Returns:
            List các tuple (nhãn, điểm tin cậy) đã được sắp xếp
        """
        if not self.model:
            self.logger.error("YAMNet model chưa được tải!")
            return None
            
        try:
            # Xử lý đầu vào là file path hoặc waveform
            if isinstance(audio_data, str):
                waveform, sr = librosa.load(audio_data, sr=sample_rate, mono=True)
            else:
                waveform = audio_data
                sr = sample_rate
                
            # Tiền xử lý waveform
            padded_waveform = self._pad_waveform(waveform)
            waveform_tensor = tf.squeeze(tf.constant(padded_waveform, dtype=tf.float32))

            # Dự đoán
            scores, embeddings, spectrogram = self.model(waveform_tensor)
            scores = scores.numpy()[0]  # Lấy frame đầu tiên

            # Lấy chỉ số của nhãn có xác suất cao nhất
            top_index = np.argmax(scores)
            best_label = self.labels[top_index] if self.labels else f"Label ID {top_index}"
            best_score = scores[top_index]

            if best_score >= threshold:
                return [(best_label, best_score)]  # Trả về 1 nhãn nếu đạt ngưỡng

            # Nếu không có nhãn nào ≥ threshold, lấy top_n nhãn có xác suất cao nhất
            top_indices = np.argsort(scores)[-top_n:][::-1]
            return [(self.labels[i] if self.labels else f"Label ID {i}", scores[i]) 
                    for i in top_indices]
                    
        except Exception as e:
            self.logger.error(f"Lỗi khi dự đoán âm thanh: {str(e)}")
            return None