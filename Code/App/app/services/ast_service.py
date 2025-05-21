import os
import argparse
import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
import queue
import datetime
import json
from collections import deque
import tempfile
import torchaudio
import torchaudio.transforms as T
from torch.cuda.amp import autocast

# Nhập thư viện xử lý âm thanh
try:
    import sounddevice as sd
    import soundfile as sf
    import scipy.io.wavfile as wav
    import scipy.signal
except ImportError:
    print("Cần cài đặt các thư viện xử lý âm thanh: sounddevice, soundfile, scipy")
    print("Sử dụng: pip install sounddevice soundfile scipy")
    exit(1)

# Tắt warnings
warnings.filterwarnings("ignore")


# Đường dẫn cố định đến model
MODEL_PATH = "D:\\NCKH\\server\\models\\ast\\ast_fsd50k_i2s_model.pt"
# Đường dẫn đến file nhãn
LABELS_FILE = "D:\\NCKH\\server\\models\\ast\\fsd50k_labels.json"

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
# Định nghĩa các lớp và hàm từ inference.py
#-----------------------------------------------------------------------------

# Các tham số mặc định của mô hình
class ModelConfig:
    # Các tham số chính
    SAMPLE_RATE = 16000
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 256
    MAX_AUDIO_LENGTH = SAMPLE_RATE * 5  # 5 giây
    
    # Các tham số preprocessing cho I2S
    I2S_BOOST_FREQ = True
    I2S_HIGH_PASS = 80
    I2S_NORMALIZATION = True
    I2S_PREEMPHASIS = 0.97
    
    # Inference parameters
    MIN_CONFIDENCE_THRESHOLD = 0.3
    MAX_DETECTIONS = 5

def identify_model_type(model_path):
    """Xác định loại model (training.py hoặc continue_training.py)"""
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Kiểm tra cấu trúc checkpoint để xác định loại model
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint and 'ESC50_NUM_CLASSES' in checkpoint['config']:
                return "ESC50 (continue_training.py)"
            elif 'model_config' in checkpoint and 'mixup_prob' in checkpoint['model_config']:
                return "FSD50K (training.py)"
            elif 'model_state_dict' in checkpoint:
                # Kiểm tra kích thước output layer
                if 'mlp_head.1.weight' in checkpoint['model_state_dict']:
                    output_shape = checkpoint['model_state_dict']['mlp_head.1.weight'].shape
                    if output_shape[0] == 50:
                        return "ESC50 (continue_training.py)"
                    else:
                        return "FSD50K (training.py)"
        
        # Trường hợp không xác định được
        return "Unknown"
    except Exception as e:
        print(f"Không thể xác định loại model: {e}")
        return "Unknown"

class ASTService:
    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('ast') if hasattr(config, 'get_logger') else None
        self.model_path = MODEL_PATH
        self.labels_file = LABELS_FILE
        self.model = None
        self.idx_to_label = {}  # Thay đổi từ self.labels thành self.idx_to_label
        if self.logger:
            self.logger.info("Đã khởi tạo AST service")

    def load_model(self):
        # Load model AST và khởi tạo recognizer

        if self.logger:
            self.logger.info(f"Đang tải model từ {os.path.basename(self.model_path)}")
        else:
            print(f"Đang tải model từ {os.path.basename(self.model_path)}")
            
        try:
            model_type = identify_model_type(MODEL_PATH)
            print(f"Đang tải model loại: {model_type} từ {os.path.basename(MODEL_PATH)}")

            # Tải checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            num_classes = None
        
            # Nếu đã có model
            if 'model' in checkpoint:
                model = checkpoint['model']
                # Chuyển model sang device
                model = model.to(device)
                model.eval()
                return model, checkpoint.get('idx_to_label', {})
            
            # Xác định số lớp từ state dict nếu không có model trực tiếp
            if 'model_state_dict' in checkpoint:
                # Tìm số lớp từ output layer
                output_layers = ['mlp_head.1.weight', 'fc.weight', 'head.weight']
                for layer_name in output_layers:
                    if layer_name in checkpoint['model_state_dict']:
                        weight_shape = checkpoint['model_state_dict'][layer_name].shape
                        num_classes = weight_shape[0]
                        print(f"Phát hiện model với {num_classes} lớp")
                        break

                if num_classes is None:
                    num_classes = 527  # FSD50K mặc định
                    print(f"Không xác định được số lớp, sử dụng mặc định: {num_classes}")

                # Tạo model
                from torch import nn

                class PatchEmbed(nn.Module):
                    def __init__(self, in_channels=1, embed_dim=768, patch_size=16):
                        super().__init__()
                        self.proj = nn.Conv2d(
                            in_channels, 
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size
                        )
                
                    def forward(self, x):
                        x = self.proj(x)
                        x = x.flatten(2)
                        x = x.transpose(1, 2)
                        return x
                    
                class AST(nn.Module):
                    def __init__(self, cfg, num_classes):
                        super().__init__()
                    
                        # Parameters
                        input_channels = 1
                        patch_size = 16
                        dim = 768
                        depth = 12
                        heads = 12
                        mlp_dim = 3072
                        
                        # Calculate number of patches
                        freq_patches = cfg.N_MELS // patch_size
                        time_patches = (cfg.MAX_AUDIO_LENGTH // cfg.HOP_LENGTH + 1) // patch_size
                        num_patches = freq_patches * time_patches
                        
                        # Define layers
                        self.patch_embed = PatchEmbed(
                            in_channels=input_channels,
                            embed_dim=dim,
                            patch_size=patch_size
                        )
                        
                        # Class token
                        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
                        
                        # Position embedding
                        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
                        
                        # Transformer encoder
                        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=dim,
                            nhead=heads,
                            dim_feedforward=mlp_dim,
                            dropout=0.1,
                            activation='gelu',
                            batch_first=True,
                            norm_first=True
                        )
                        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
                        
                        # MLP head
                        self.mlp_head = nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, num_classes)
                        )
                    
                    def forward(self, x):
                        B = x.shape[0]
                    
                        # Patch embedding
                        x = self.patch_embed(x)
                        
                        # Add class token
                        cls_tokens = self.cls_token.expand(B, -1, -1)
                        x = torch.cat((cls_tokens, x), dim=1)
                        
                        # Add positional embedding
                        x = x + self.pos_embed
                        
                        # Apply transformer
                        x = self.transformer(x)
                        
                        # Extract class token for classification
                        x = x[:, 0]
                        
                        # MLP head
                        x = self.mlp_head(x)
                        
                        return x

                try:
                    model = AST(ModelConfig, num_classes)
                
                    print("Đang tải state dictionary cho model...")
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                    # Chuyển model sang device
                    model = model.to(device)
                    model.eval()
                    print("Model đã được tải thành công từ state dict")

                except Exception as e:
                    print(f"Lỗi khi tải state dict cho model: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, {}
            else:
                print("Không tìm thấy model hoặc model_state_dict trong checkpoint!")
                return None, {}
            
            # Tải mapping nhãn
            idx_to_label = checkpoint.get('idx_to_label', {})

            # Tải nhãn từ file nếu không có trong checkpoint
            if not idx_to_label and os.path.exists(LABELS_FILE):
                try:
                    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                        labels_data = json.load(f)
                        # Kiểm tra cấu trúc file
                        if 'idx_to_label' in labels_data:
                            idx_to_label = labels_data['idx_to_label']
                        else:
                            idx_to_label = labels_data
                    
                    # Chuyển key từ string sang int nếu cần
                    if all(isinstance(k, str) for k in idx_to_label.keys()):
                        idx_to_label = {int(k): v for k, v in idx_to_label.items()}
                    
                    print(f"Đã tải {len(idx_to_label)} nhãn từ file")
                except Exception as e:
                    print(f"Lỗi khi tải file nhãn: {e}")

            self.model = model
            self.idx_to_label = idx_to_label  # Lưu trữ idx_to_label vào thuộc tính
            return self
        
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            import traceback
            traceback.print_exc()
            return None, {}

    def predict(self, audio_path, use_tta=False, confidence_threshold=0.3, labels_mapping_file=None):
        """
        Dự đoán âm thanh từ file sử dụng model AST
        """

        if self.model is None:
            print("Model chưa được tải, không thể dự đoán!")
            return []
    
        # Kiểm tra file âm thanh tồn tại
        if not os.path.exists(audio_path):
            print(f"Không tìm thấy file âm thanh: {audio_path}")
            return []

            
        # Tải và tiền xử lý âm thanh
        waveform, sample_rate = torchaudio.load(audio_path)
            
        # Chuyển sang mono nếu stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample nếu cần
        if sample_rate != ModelConfig.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, ModelConfig.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Pre-processing
        # Các bước xử lý như normalize, high-pass filter, pre-emphasis được 
        # thực hiện bên trong model nên không cần thực hiện ở đây
            
        # Chia thành các đoạn 5 giây với overlap 50% nếu audio dài hơn 5 giây
        segment_length = ModelConfig.MAX_AUDIO_LENGTH
        hop_length = segment_length // 2  # 50% overlap
            
        segments = []
        if waveform.shape[1] > segment_length:
            # Chia thành các đoạn
            for start in range(0, waveform.shape[1], hop_length):
                end = min(start + segment_length, waveform.shape[1])
                segment = waveform[:, start:end]
                    
                # Pad nếu segment ngắn hơn segment_length
                if segment.shape[1] < segment_length:
                    segment = torch.nn.functional.pad(segment, (0, segment_length - segment.shape[1]))
                    
                segments.append(segment)
        else:
            # Pad nếu cần
            if waveform.shape[1] < segment_length:
                waveform = torch.nn.functional.pad(waveform, (0, segment_length - waveform.shape[1]))
            segments = [waveform]
            
        # Dự đoán cho từng đoạn
        all_segment_probs = []
            
        for i, segment in enumerate(segments):
            # Tạo mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=ModelConfig.SAMPLE_RATE,
                n_fft=ModelConfig.N_FFT,
                hop_length=ModelConfig.HOP_LENGTH,
                n_mels=ModelConfig.N_MELS
            )
                
            mel_spec = mel_transform(segment)
                
            # Chuyển sang decibels
            mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                
            # Chuẩn hóa
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
                
            # Dự đoán với model
            with torch.no_grad():
                with autocast(enabled=True):
                    mel = mel.unsqueeze(0).to(device)  # Thêm batch dimension
                    output = self.model(mel)
                        
                    # Áp dụng sigmoid để chuyển sang xác suất
                    segment_probs = torch.sigmoid(output)[0].cpu().numpy()
                    all_segment_probs.append(segment_probs)
            
        # Kết hợp dự đoán từ các đoạn
        probs = np.mean(np.array(all_segment_probs), axis=0)
            
        # Chọn các lớp có confidence cao
        high_conf_idx = np.where(probs > confidence_threshold)[0]
            
        if len(high_conf_idx) == 0:
            print(f"Không có lớp nào có độ tin cậy > {confidence_threshold}")
            return []
            
        # Sắp xếp theo độ tin cậy
        high_conf_idx = high_conf_idx[np.argsort(probs[high_conf_idx])[::-1]]
            
        # Giới hạn số lượng lớp
        high_conf_idx = high_conf_idx[:ModelConfig.MAX_DETECTIONS]
            
        # Tạo danh sách kết quả
        predictions = []
        for idx in high_conf_idx:
            label = self.idx_to_label.get(idx, f"Class {idx}")
            confidence = float(probs[idx])
            predictions.append((label, confidence))
            
        return predictions
    
    def record_and_predict(self, duration=5, fs=16000):
        """Thu âm từ microphone và dự đoán liên tục"""
        if self.model is None:
            print("Model chưa được load!")
            return

        print(f"Sử dụng model đã load sẵn")

        # Tạo thư mục tạm để lưu các đoạn âm thanh
        temp_dir = tempfile.mkdtemp()
        print(f"Lưu các đoạn âm thanh tạm thời vào: {temp_dir}")

        # Tạo hàng đợi để lưu các đoạn âm thanh trong 1 phút (60s / 5s = 12 đoạn)
        audio_queue = deque(maxlen=12)

        # Biến cờ để dừng thu âm
        stop_recording = threading.Event()

        # Biến đếm số đoạn âm thanh đã thu
        segment_count = 0

        # Hàm tiền xử lý âm thanh tương tự pipeline training
        def preprocess_audio(audio_data, sample_rate):
            if not isinstance(audio_data, torch.Tensor):
                audio_data = torch.tensor(audio_data, dtype=torch.float32)
            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)
            if ModelConfig.I2S_NORMALIZATION:
                max_val = torch.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max_val
            if ModelConfig.I2S_HIGH_PASS > 0:
                nyquist = sample_rate // 2
                normal_cutoff = ModelConfig.I2S_HIGH_PASS / nyquist
                b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
                wave_np = audio_data.cpu().numpy().squeeze().copy()
                filtered = scipy.signal.filtfilt(b, a, wave_np)
                audio_data = torch.FloatTensor(filtered.copy()).unsqueeze(0)
            if ModelConfig.I2S_PREEMPHASIS > 0:
                wave = audio_data.squeeze().clone()
                first_sample = wave[:1].clone()
                rest = wave[1:] - ModelConfig.I2S_PREEMPHASIS * wave[:-1]
                audio_data = torch.cat([first_sample, rest]).unsqueeze(0)
            return audio_data

        # Hàm dự đoán âm thanh trực tiếp từ tensor
        def predict_audio_tensor(audio_tensor, sample_rate):
            try:
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=ModelConfig.SAMPLE_RATE,
                    n_fft=ModelConfig.N_FFT,
                    hop_length=ModelConfig.HOP_LENGTH,
                    n_mels=ModelConfig.N_MELS
                )
                mel_spec = mel_transform(audio_tensor)
                mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                mel = (mel - mel.mean()) / (mel.std() + 1e-8)

                with torch.no_grad():
                    with autocast(enabled=True):
                        mel = mel.unsqueeze(0).to(device)
                        output = self.model(mel)
                        probs = torch.sigmoid(output)[0].cpu().numpy()

                high_conf_idx = np.where(probs > ModelConfig.MIN_CONFIDENCE_THRESHOLD)[0]
                if len(high_conf_idx) == 0:
                    return []
                high_conf_idx = high_conf_idx[np.argsort(probs[high_conf_idx])[::-1]]
                high_conf_idx = high_conf_idx[:ModelConfig.MAX_DETECTIONS]

                predictions = []
                for idx in high_conf_idx:
                    label = self.idx_to_label.get(idx, f"Class {idx}")
                    confidence = float(probs[idx])
                    predictions.append((label, confidence))

                return predictions

            except Exception as e:
                print(f"Lỗi trong quá trình dự đoán: {e}")
                return []

        print("\nBắt đầu thu âm từ microphone...")
        print("Nhấn Ctrl+C để dừng.")

        AUDIO_THRESHOLD = 0.0009

        try:
            while True:
                print(f"\n--- Thu âm đoạn {segment_count+1} trong {duration} giây... ---")
                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()

                audio_energy = np.mean(np.abs(audio_data))
                if audio_energy < AUDIO_THRESHOLD:
                    print(f"Không phát hiện âm thanh có ý nghĩa (năng lượng: {audio_energy:.6f} < ngưỡng: {AUDIO_THRESHOLD})")
                    segment_count += 1
                    continue

                audio_tensor = preprocess_audio(audio_data, fs)

                print(f"\n--- Đang dự đoán cho đoạn {segment_count+1} ---")
                predictions = predict_audio_tensor(audio_tensor, fs)

                if predictions:
                    print("\nKết quả nhận dạng:")
                    for label, prob in predictions:
                        print(f"  {label}: {prob:.4f}")
                else:
                    print("Không phát hiện âm thanh nào!")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = os.path.join(temp_dir, f"segment_{segment_count}_{timestamp}.wav")
                processed_audio = audio_tensor.squeeze().numpy()
                sf.write(temp_file, processed_audio, fs)
                audio_queue.append(temp_file)

                if len(audio_queue) == 12 and segment_count >= 12:
                    old_file = audio_queue[0]
                    if os.path.exists(old_file):
                        try:
                            print(f"Xóa đoạn âm thanh cũ: {os.path.basename(old_file)}")
                            os.remove(old_file)
                        except Exception as e:
                            print(f"Không thể xóa file {os.path.basename(old_file)}: {e}")

                segment_count += 1

        except KeyboardInterrupt:
            print("\nDừng thu âm...")
        finally:
            try:
                for audio_file in audio_queue:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                os.rmdir(temp_dir)
                print(f"Đã xóa thư mục tạm: {temp_dir}")
            except Exception as e:
                print(f"Lỗi khi xóa thư mục tạm: {e}")
