import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import json
from sklearn.metrics import average_precision_score
import time
import random
from torch.cuda.amp import autocast, GradScaler
import math

# Ignore warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    # Paths - Windows style
    FSD50K_ROOT = "C:\\Users\\princ\\Downloads\\data\\FSD50K"  # Update with your FSD50K path
    DEV_AUDIO = os.path.join(FSD50K_ROOT, "FSD50K.dev_audio")
    EVAL_AUDIO = os.path.join(FSD50K_ROOT, "FSD50K.eval_audio")
    GROUND_TRUTH = os.path.join(FSD50K_ROOT, "FSD50K.ground_truth")
    METADATA = os.path.join(FSD50K_ROOT, "FSD50K.ground_truth")
    
    # Training parameters
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    NUM_WORKERS = 8
    LEARNING_RATE = 2e-4  # Điều chỉnh learning rate 
    WEIGHT_DECAY = 2e-5   # Điều chỉnh weight decay
    EPOCHS = 300          # Đủ epochs để đạt độ chính xác cao
    GRAD_ACCUMULATION_STEPS = 4
    
    # Thêm các tham số mới cho training
    EARLY_STOPPING_PATIENCE = 30  # Tăng patience để đảm bảo hội tụ tối ưu
    LR_REDUCE_FACTOR = 0.7        # Điều chỉnh hệ số giảm learning rate
    LR_REDUCE_PATIENCE = 5        # Số epochs không cải thiện trước khi giảm LR
    MIN_LR = 1e-6                 # Learning rate tối thiểu
    
    # Model parameters - tối ưu cho I2S
    SAMPLE_RATE = 16000           # Tần số lấy mẫu tiêu chuẩn cho I2S
    CLIP_DURATION = 5             # Tăng độ dài clip để nhận dạng tốt hơn
    MAX_AUDIO_LENGTH = SAMPLE_RATE * CLIP_DURATION
    N_MELS = 128                  # Số mel bands
    N_FFT = 1024                  # FFT size
    HOP_LENGTH = 256              # Hop length
    PATCH_SIZE = 16
    DIM = 768                     # Tăng kích thước để có độ chính xác cao
    DEPTH = 12                    # Tăng số lớp 
    HEADS = 12                    # Tăng số heads
    MLP_DIM = 3072                # Tăng MLP dim
    
    # Tối ưu cho tín hiệu I2S
    I2S_BOOST_FREQ = True         # Tăng cường tần số cao cho tín hiệu I2S
    I2S_HIGH_PASS = 80            # Tần số cắt thấp (Hz) để loại bỏ nhiễu tần thấp
    I2S_NORMALIZATION = True      # Chuẩn hóa biên độ tín hiệu I2S
    I2S_PREEMPHASIS = 0.97        # Pre-emphasis để tăng cường tần số cao
    
    # Augmentation parameters - Tối ưu cho I2S
    TIME_MASK_PARAM = 60          # Time masking
    FREQ_MASK_PARAM = 20          # Freq masking
    TIME_STRETCH_RANGE = (0.9, 1.1)  # Thu hẹp phạm vi để giữ đặc tính âm thanh
    PITCH_SHIFT_RANGE = (-2, 2)      # Thu hẹp phạm vi pitch shift
    NOISE_SNR_RANGE = (10, 25)       # Điều chỉnh SNR cho nhiễu I2S
    
    # Loss parameters
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = 0.25
    
    # Mixed precision training
    USE_AMP = True
    
    # Model saving
    MODEL_SAVE_PATH = "ast_fsd50k_i2s_model.pt"
    CHECKPOINT_DIR = "checkpoints"
    
    # Debug mode
    DEBUG = False
    DEBUG_SAMPLES = 100
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inference parameters
    MIN_CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng tin cậy tối thiểu
    MAX_DETECTIONS = 5  # Số lượng âm thanh tối đa phát hiện được
    OVERLAP_THRESHOLD = 0.5  # Ngưỡng chồng lấn giữa các âm thanh
    NOISE_FLOOR = 0.1

# Create config object
cfg = Config()

# Create checkpoint directory if it doesn't exist
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

# Utility function to load metadata
def load_metadata():
    # Load vocabulary from the metadata
    try:
        vocab_path = os.path.join(cfg.METADATA, "vocabulary.csv")
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(cfg.GROUND_TRUTH, "vocabulary.csv")
            print(f"Không tìm thấy file tại {os.path.join(cfg.METADATA, 'vocabulary.csv')}")
            print(f"Thử dùng đường dẫn thay thế: {vocab_path}")
        
        with open(vocab_path, "r") as f:
            vocabulary = pd.read_csv(f, header=None)
        labels = vocabulary[1].tolist()
        
        # Create label to index mapping
        label_to_idx = {label: i for i, label in enumerate(labels)}
        idx_to_label = {i: label for i, label in enumerate(labels)}
        
        # Load development set metadata
        dev_csv = os.path.join(cfg.GROUND_TRUTH, "dev.csv")
        dev_meta = pd.read_csv(dev_csv)
        
        # Load evaluation set metadata
        eval_csv = os.path.join(cfg.GROUND_TRUTH, "eval.csv")
        eval_meta = pd.read_csv(eval_csv)
        
        # Loại bỏ tất cả các print statement in thông tin về metadata
        # để tránh in lặp lại trong quá trình training
        
        return labels, label_to_idx, idx_to_label, dev_meta, eval_meta
    except Exception as e:
        print(f"Error loading metadata: {e}")
        raise e

# Load metadata
def get_metadata():
    try:
        labels, label_to_idx, idx_to_label, dev_meta, eval_meta = load_metadata()
        return labels, label_to_idx, idx_to_label, dev_meta, eval_meta
    except Exception as e:
        print(f"Error loading metadata: {e}")
        print("Please check the path to FSD50K dataset and update Config.FSD50K_ROOT")
        exit(1)

labels, label_to_idx, idx_to_label, dev_meta, eval_meta = get_metadata()
num_classes = len(labels)

# Thêm các augmentation mới
class TimeStretch(nn.Module):
    def __init__(self, stretch_range):
        super().__init__()
        self.stretch_range = stretch_range
        
    def forward(self, waveform):
        try:
            # Cách tiếp cận an toàn hơn với librosa
            import librosa
            import numpy as np
            
            # Lấy stretch factor
            stretch_factor = torch.FloatTensor(1).uniform_(*self.stretch_range)[0].item()
            
            # Chuyển sang numpy
            waveform_np = waveform.cpu().numpy().squeeze().copy()
            
            # Áp dụng time stretch với librosa
            stretched = librosa.effects.time_stretch(waveform_np, rate=stretch_factor)
            
            # Đảm bảo độ dài không đổi so với input
            if len(stretched) < waveform.shape[1]:
                # Pad nếu ngắn hơn
                pad_len = waveform.shape[1] - len(stretched)
                stretched = np.pad(stretched, (0, pad_len))
            elif len(stretched) > waveform.shape[1]:
                # Cắt bớt nếu dài hơn
                stretched = stretched[:waveform.shape[1]]
                
            # Chuyển về torch và giữ kích thước như cũ
            return torch.FloatTensor(stretched).unsqueeze(0)
        except Exception as e:
            print(f"Lỗi time stretch: {e}")
            return waveform  # Trả về input ban đầu nếu lỗi

class PitchShift(nn.Module):
    def __init__(self, shift_range):
        super().__init__()
        self.shift_range = shift_range
        
    def forward(self, waveform):
        # Đơn giản hóa - chỉ thực hiện khi có thể import librosa
        try:
            import librosa
            shift_steps = random.randint(*self.shift_range)
            if shift_steps == 0:
                return waveform
            
            # Convert to numpy for librosa
            audio_np = waveform.numpy().squeeze()
            shifted = librosa.effects.pitch_shift(audio_np, sr=16000, n_steps=shift_steps)
            return torch.FloatTensor(shifted).unsqueeze(0)
        except:
            return waveform

class AddBackgroundNoise(nn.Module):
    def __init__(self, snr_range):
        super().__init__()
        self.snr_range = snr_range
    
    def forward(self, waveform):
        # Tạo noise với SNR ngẫu nhiên
        snr = random.uniform(*self.snr_range)
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        
        # Tạo white noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

# Thêm xử lý cho tín hiệu I2S

# Dataset class for FSD50K with enhanced augmentations
class FSD50KDataset(Dataset):
    def __init__(self, metadata, audio_dir, cfg, augment=False):
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.cfg = cfg
        self.augment = augment
        
        # Thêm audio preprocessor
        self.preprocessor = AudioPreprocessor(cfg)
        
       # Giảm số lượng mẫu khi debug
        if cfg.DEBUG:
            self.metadata = self.metadata.iloc[:cfg.DEBUG_SAMPLES]
            print(f"DEBUG mode: Sử dụng {len(self.metadata)} mẫu")
        
        # Mel spectrogram transformation
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.SAMPLE_RATE,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
        )
        
        # Enhanced augmentations
        if augment:
            self.time_mask = T.TimeMasking(time_mask_param=cfg.TIME_MASK_PARAM)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.FREQ_MASK_PARAM)
            
            # Waveform augmentations
            try:
                self.time_stretch = TimeStretch(cfg.TIME_STRETCH_RANGE)
                self.pitch_shift = PitchShift(cfg.PITCH_SHIFT_RANGE)
                self.add_noise = AddBackgroundNoise(cfg.NOISE_SNR_RANGE)
                self.has_advanced_augs = True
                print("Advanced augmentations enabled")
            except:
                self.has_advanced_augs = False
                print("Advanced augmentations not available")
        
        # Kiểm tra các file audio
        if cfg.DEBUG:
            found = 0
            missing = 0
            for idx, row in self.metadata.iterrows():
                fname = f"{row['fname']}.wav"
                audio_path = os.path.join(self.audio_dir, fname)
                if os.path.exists(audio_path):
                    found += 1
                else:
                    missing += 1
                    if missing < 10:  # Chỉ in 10 file đầu tiên bị thiếu
                        print(f"Missing audio file: {audio_path}")
            
            print(f"Checked audio files: {found} found, {missing} missing")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get audio file path and labels
        row = self.metadata.iloc[idx]
        fname = f"{row['fname']}.wav"
        audio_path = os.path.join(self.audio_dir, fname)
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Tạo bản sao của waveform để tránh lỗi stride
            waveform = waveform.clone()
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.cfg.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.cfg.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Ensure waveform is valid
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                print(f"WARNING: File {fname} contains NaN or Inf values. Replacing with zeros.")
                waveform = torch.zeros_like(waveform)
            
            # Áp dụng xử lý cho tín hiệu I2S với xử lý lỗi cẩn thận
            try:
                waveform = self.preprocessor.process(waveform)
            except Exception as e:
                print(f"Lỗi xử lý I2S cho {fname}: {e}")
                # Vẫn giữ waveform gốc nếu xử lý thất bại
            
            # Apply waveform augmentations - với xử lý lỗi
            if self.augment and hasattr(self, 'has_advanced_augs') and self.has_advanced_augs:
                # Apply augmentation với xác suất 50%
                # Luôn dùng waveform.clone() để tránh lỗi stride
                aug_waveform = waveform.clone()
                
                if random.random() < 0.5:
                    try:
                        aug_waveform = self.add_noise(aug_waveform)
                    except Exception as e:
                        print(f"Lỗi khi thêm noise cho {fname}: {e}")
                
                if random.random() < 0.3:
                    try:
                        aug_waveform = self.time_stretch(aug_waveform)
                    except Exception as e:
                        print(f"Lỗi khi time stretch cho {fname}: {e}")
                
                if random.random() < 0.3:
                    try:
                        aug_waveform = self.pitch_shift(aug_waveform)
                    except Exception as e:
                        print(f"Lỗi khi pitch shift cho {fname}: {e}")
                    
                # Chỉ cập nhật waveform nếu không có lỗi
                if not (torch.isnan(aug_waveform).any() or torch.isinf(aug_waveform).any()):
                    waveform = aug_waveform
            
            # Pad or truncate to desired length
            if waveform.shape[1] < self.cfg.MAX_AUDIO_LENGTH:
                # Pad
                padding = self.cfg.MAX_AUDIO_LENGTH - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            else:
                # Random crop or center crop
                if self.augment:
                    max_start = waveform.shape[1] - self.cfg.MAX_AUDIO_LENGTH
                    start = random.randint(0, max_start)
                else:
                    start = (waveform.shape[1] - self.cfg.MAX_AUDIO_LENGTH) // 2
                
                # Clone để tránh lỗi stride
                waveform = waveform[:, start:start + self.cfg.MAX_AUDIO_LENGTH].clone()
            
            # Convert to mel spectrogram
            try:
                mel = self.mel_spec(waveform)
                
                # Tăng cường tần số cao nếu cần với xử lý lỗi
                try:
                    mel = self.preprocessor.boost_high_freq(mel)
                except Exception as e:
                    print(f"Lỗi boost high freq cho {fname}: {e}")
                
                # Apply data augmentation nâng cao nếu cần
                if self.augment:
                    # Sử dụng SpecAugment nâng cao
                    aug_mel = mel.clone()  # Tạo bản sao để tránh lỗi
                    
                    if random.random() < 0.7:
                        try:
                            # Áp dụng augment nâng cao - thay thế time_mask và freq_mask
                            aug_mel = self.preprocessor.advanced_spec_augment(aug_mel)
                        except Exception as e:
                            print(f"Lỗi khi áp dụng advanced_spec_augment cho {fname}: {e}")
                            # Fallback to original masking
                            try:
                                aug_mel = self.time_mask(aug_mel)
                                aug_mel = self.freq_mask(aug_mel)
                            except Exception as e2:
                                print(f"Lỗi khi áp dụng masking gốc cho {fname}: {e2}")
                    
                    # Chỉ sử dụng kết quả augment nếu hợp lệ
                    if not (torch.isnan(aug_mel).any() or torch.isinf(aug_mel).any()):
                        mel = aug_mel
                
                # Convert to decibels
                mel = T.AmplitudeToDB()(mel)
                
                # Normalize - xử lý an toàn để tránh NaN
                try:
                    mean = mel.mean()
                    std = mel.std() + 1e-8  # Tránh chia cho 0
                    
                    if torch.isnan(mean) or torch.isnan(std) or torch.isinf(mean) or torch.isinf(std):
                        print(f"WARNING: NaN/Inf trong mel spec cho {fname}. Dùng giá trị mặc định.")
                        mel = torch.zeros_like(mel)
                    else:
                        mel = (mel - mean) / std
                except Exception as e:
                    print(f"Lỗi khi chuẩn hóa mel spec cho {fname}: {e}")
                    mel = torch.zeros_like(mel)
                
                # Kiểm tra lần cuối mel spec có hợp lệ không
                if torch.isnan(mel).any() or torch.isinf(mel).any():
                    print(f"WARNING: Mel spec cuối cùng có NaN/Inf cho {fname}. Dùng zeros.")
                    mel = torch.zeros_like(mel)
                
                # Get multi-hot encoded labels
                labels_list = row['labels'].split(',')
                target = torch.zeros(num_classes)
                for label in labels_list:
                    label_idx = label_to_idx[label.strip()]
                    target[label_idx] = 1.0
                
                return mel, target
            except Exception as e:
                print(f"Lỗi khi tạo mel spectrogram cho {fname}: {e}")
                raise  # Re-raise để có thể xử lý ở mức cao hơn
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a placeholder in case of error
            mel = torch.zeros(1, self.cfg.N_MELS, self.cfg.MAX_AUDIO_LENGTH // self.cfg.HOP_LENGTH + 1)
            target = torch.zeros(num_classes)
            return mel, target


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        
        # Calculate positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return x


# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=384, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, F, T)
        x = self.proj(x)  # (B, embed_dim, F//patch_size, T//patch_size)
        x = x.flatten(2)  # (B, embed_dim, F//patch_size * T//patch_size)
        x = x.transpose(1, 2)  # (B, F//patch_size * T//patch_size, embed_dim)
        return x


# Audio Spectrogram Transformer
class AST(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        
        # Parameters
        input_channels = 1
        patch_size = cfg.PATCH_SIZE
        dim = cfg.DIM
        depth = cfg.DEPTH
        heads = cfg.HEADS
        mlp_dim = cfg.MLP_DIM
        
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
        
        # Calculate number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params / 1e6:.2f}M parameters")
    
    def forward(self, x):
        # x: (B, C, F, T)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Extract class token for classification
        x = x[:, 0]
        
        # MLP head
        x = self.mlp_head(x)
        
        return x


# Focal Loss cải tiến cho multi-label classification
class EnhancedFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, alpha_pos=0.25, alpha_neg=0.75, reduction='mean'):
        super(EnhancedFocalLoss, self).__init__()
        self.gamma_pos = gamma_pos  # Gamma cho positive samples
        self.gamma_neg = gamma_neg  # Gamma cao hơn cho negative samples
        self.alpha_pos = alpha_pos  # Alpha cho positive samples
        self.alpha_neg = alpha_neg  # Alpha cao hơn cho negative samples
        self.reduction = reduction
        
    def forward(self, inputs, targets, weights=None):
        # Áp dụng sigmoid
        probs = torch.sigmoid(inputs)
        
        # Tính Focal weights riêng cho positive và negative samples
        # Đối với I2S, negative samples cần được điều chỉnh mạnh hơn
        focal_pos = self.alpha_pos * (1 - probs).pow(self.gamma_pos) * targets
        focal_neg = self.alpha_neg * probs.pow(self.gamma_neg) * (1 - targets)
        focal_weights = focal_pos + focal_neg
        
        # BCE loss
        bce_loss = F.sss(inputs, targets, reduction='none')
        
        # Áp dụng focal weights
        focal_loss = focal_weights * bce_loss
        
        # Áp dụng class weights nếu có
        if weights is not None:
            weights = weights.to(focal_loss.device)
            focal_loss = focal_loss * weights.view(1, -1)
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


# Function to add AST class to safe globals after it's defined
def register_ast_for_loading():
    try:
        # Check if AST is defined
        if 'AST' in globals():
            torch.serialization.add_safe_globals([AST])
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not register AST for loading: {e}")
        return False


# Thêm hàm Label Smoothing trước hàm mixup_data
def smooth_targets(targets, smoothing=0.05):
    """Áp dụng label smoothing cho multi-label classification để cải thiện mAP"""
    # Làm mịn các nhãn 1 và giữ cấu trúc sparse của nhãn multi-label
    targets = targets * (1 - smoothing) + smoothing / targets.sum(dim=1, keepdim=True).clamp(min=1)
    return targets

# Thêm mới: MixUp augmentation cho huấn luyện
def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation đặc biệt cho dữ liệu âm thanh I2S"""
    batch_size = x.size(0)
    if alpha > 0:
        # Tạo hệ số trộn từ phân phối Beta
        lam = np.random.beta(alpha, alpha, batch_size)
        
        # Đảm bảo lam không quá nhỏ để vẫn giữ lại đủ thông tin 
        lam = np.maximum(lam, 1-lam)
        
        # Chuyển sang tensor và đặt trên cùng device
        lam = torch.from_numpy(lam).float().to(x.device)
        
        # Tạo các chỉ số xáo trộn
        index = torch.randperm(batch_size).to(x.device)
        
        # Reshape lam để nhân với tensor 4D (batch, channel, freq, time)
        lam_expanded = lam.view(-1, 1, 1, 1)
        
        # Trộn dữ liệu input
        mixed_x = lam_expanded * x + (1 - lam_expanded) * x[index]
        
        # Trộn nhãn (cho multi-label - phù hợp với FSD50K)
        lam_expanded_y = lam.view(-1, 1)
        mixed_y = lam_expanded_y * y + (1 - lam_expanded_y) * y[index]
        
        return mixed_x, mixed_y
    else:
        return x, y

# Định nghĩa lịch trình học tập tốt hơn với warm restarts
def get_cosine_schedule_with_warmup_restarts(optimizer, num_warmup_steps, num_training_steps, 
                                           num_cycles=2, min_factor=0.1, last_epoch=-1):
    """
    Lịch trình học cosine với warmup và restarts
    num_warmup_steps: số bước warmup
    num_training_steps: tổng số bước huấn luyện
    num_cycles: số chu kỳ cosine
    min_factor: hệ số learning rate tối thiểu
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Progress từ warmup đến total (0 đến 1)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Tính số vòng hiện tại
        cycle_progress = progress * num_cycles
        cycle = int(cycle_progress)
        cycle_progress = cycle_progress - cycle  # phần thập phân
        
        # Cosine decay với restart
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        
        # Thêm tính năng tăng dần min_lr sau mỗi chu kỳ
        cycle_min_factor = min_factor + (1.0 - min_factor) * (cycle / num_cycles)
        
        # Đảm bảo không bao giờ xuống quá thấp
        return max(min_factor, cycle_min_factor + (1.0 - cycle_min_factor) * cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# Thêm hàm để phân chia các layers của transformer encoder cho discriminative fine-tuning
def get_transformer_layer_params(transformer, num_layers):
    """Phân chia các tham số của transformer thành các nhóm cho discriminative fine-tuning"""
    # Chia các layer thành 3 nhóm: đầu, giữa, cuối
    early_layers_params = []
    middle_layers_params = []
    late_layers_params = []
    
    # Chia các layers của transformer
    one_third = num_layers // 3
    two_thirds = 2 * one_third
    
    for i, layer in enumerate(transformer.layers):
        if i < one_third:
            early_layers_params.extend(layer.parameters())
        elif i < two_thirds:
            middle_layers_params.extend(layer.parameters())
        else:
            late_layers_params.extend(layer.parameters())
    
    return [early_layers_params, middle_layers_params, late_layers_params]

# Main training function
def train():
    # Đặt seed một lần nữa để đảm bảo reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Bắt đầu huấn luyện mô hình nhận dạng âm thanh cho I2S với FSD50K")
    print(f"Cấu hình model: DIM={cfg.DIM}, DEPTH={cfg.DEPTH}, HEADS={cfg.HEADS}")
    print(f"Cấu hình âm thanh: SR={cfg.SAMPLE_RATE}Hz, N_MELS={cfg.N_MELS}")
    
    # Create datasets
    print("Creating datasets...")
    dev_dataset = FSD50KDataset(dev_meta, cfg.DEV_AUDIO, cfg, augment=True)
    eval_dataset = FSD50KDataset(eval_meta, cfg.EVAL_AUDIO, cfg, augment=False)
    
    # Split dev set into train and validation
    train_size = int(0.95 * len(dev_dataset))  # Tăng tỷ lệ dữ liệu train lên 95%
    val_size = len(dev_dataset) - train_size
    train_dataset, val_dataset = random_split(dev_dataset, [train_size, val_size])
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(eval_dataset)} samples")
    
    # Create data loaders with Windows-specific settings
    # Note: Windows sometimes has issues with num_workers > 0, 
    # so we'll add some error handling
    try:
        # Try with requested number of workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if cfg.NUM_WORKERS > 0 else False,  # Keep workers alive between iterations
        )
        
        # Test with a single batch
        inputs, targets = next(iter(train_loader))
        print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}")
        print(f"Target sparsity: {targets.sum()/targets.numel():.6f}")
        
        # If successful, create val and test loaders with same settings
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.EVAL_BATCH_SIZE, 
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        )
        
        test_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        )
        
    except Exception as e:
        print(f"Error with multi-processing: {e}")
        print("Falling back to single worker for Windows compatibility")
        
        # Fall back to smaller number of workers or single worker
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # No multiprocessing
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    # Training loop
    best_val_map = 0.0
    start_time = time.time()
    
    # Khởi tạo biến start_epoch từ đầu để tránh lỗi
    start_epoch = 0
    latest_epoch = 0
    checkpoint_found = False
    
    # Định nghĩa hàm lưu model an toàn
    def save_model_safely(checkpoint, path, model_desc=""):
        try:
            # Thêm thông tin phiên bản PyTorch và thời gian lưu
            checkpoint['torch_version'] = torch.__version__
            checkpoint['saved_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            checkpoint['class_weights'] = class_weights if 'class_weights' in globals() else None
            
            torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
            print(f"Saved {model_desc} to {path}")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu {model_desc}: {e}")
            # Thử lưu với _use_new_zipfile_serialization=True
            try:
                torch.save(checkpoint, path)
                print(f"Saved {model_desc} với zipfile serialization mới to {path}")
                return True
            except Exception as e2:
                print(f"Vẫn không thể lưu {model_desc}: {e2}")
                return False
    
    try:
        # Kiểm tra thư mục checkpoint
        checkpoint_found = False
        latest_epoch = 0
        latest_checkpoint = None
        model_config = None
        
        if os.path.exists(cfg.CHECKPOINT_DIR):
            checkpoint_files = os.listdir(cfg.CHECKPOINT_DIR)
            checkpoint_files = [f for f in checkpoint_files if f.startswith("ast_i2s_epoch_") and f.endswith(".pt")]
            
            if checkpoint_files:
                print("Tìm thấy các checkpoint. Đang kiểm tra checkpoint mới nhất...")
                for checkpoint_file in checkpoint_files:
                    try:
                        epoch_num = int(checkpoint_file.split("_")[3])
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            latest_checkpoint = checkpoint_file
                    except:
                        continue
                
                if latest_checkpoint:
                    print(f"Tìm thấy checkpoint mới nhất: {latest_checkpoint}")
                    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, latest_checkpoint)
                    checkpoint_found = True
        else:
            print("Thư mục checkpoint không tồn tại.")
    except Exception as e:
        print(f"Lỗi khi kiểm tra checkpoint: {e}")
        checkpoint_found = False
    
    # Create model (tạo model trước, nếu có checkpoint sẽ tải vào model này)
    print("Tạo mô hình mới...")
    model = AST(cfg, num_classes)
    model = model.to(cfg.DEVICE)
    
    # Register AST class for safe loading
    register_ast_for_loading()
    
    # Create optimizer with discriminative fine-tuning
    print("Áp dụng Discriminative Fine-tuning cho mô hình...")
    
    # Chia mô hình thành các nhóm parameters với learning rates khác nhau
    transformer_params = get_transformer_layer_params(model.transformer, cfg.DEPTH)
    
    parameters = [
        {'params': model.patch_embed.parameters(), 'lr': cfg.LEARNING_RATE * 0.1},  # Embedding layers học chậm nhất
        {'params': transformer_params[0], 'lr': cfg.LEARNING_RATE * 0.2},           # Early layers
        {'params': transformer_params[1], 'lr': cfg.LEARNING_RATE * 0.5},           # Middle layers
        {'params': transformer_params[2], 'lr': cfg.LEARNING_RATE},                 # Late layers
        {'params': model.mlp_head.parameters(), 'lr': cfg.LEARNING_RATE * 1.2}      # Head layer học nhanh nhất
    ]
    
    optimizer = optim.AdamW(
        parameters,
        lr=cfg.LEARNING_RATE,  # Base learning rate
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    
    # Tính toán tổng số bước huấn luyện điều chỉnh - đảm bảo start_epoch đã được khởi tạo
    total_steps = len(train_loader) * (cfg.EPOCHS - start_epoch) // cfg.GRAD_ACCUMULATION_STEPS
    
    # Thay thế scheduler bằng OneCycleLR để cải thiện tốc độ hội tụ
    print("Sử dụng OneCycleLR scheduler để tăng tốc độ hội tụ...")
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            cfg.LEARNING_RATE * 0.1 * 3,  # Peak learning rate cho embedding
            cfg.LEARNING_RATE * 0.2 * 3,  # Peak learning rate cho early layers
            cfg.LEARNING_RATE * 0.5 * 3,  # Peak learning rate cho middle layers
            cfg.LEARNING_RATE * 3,        # Peak learning rate cho late layers
            cfg.LEARNING_RATE * 1.2 * 3   # Peak learning rate cho head
        ],
        total_steps=total_steps,
        pct_start=0.3,                    # 30% training time là warmup phase
        div_factor=25,                    # Initial lr = max_lr/25
        final_div_factor=1000,            # Final lr = initial_lr/1000
        anneal_strategy='cos'             # Cosine annealing
    )
    
    # Thêm ReduceLROnPlateau scheduler để giảm LR khi validation không cải thiện
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=cfg.LR_REDUCE_FACTOR,
        patience=cfg.LR_REDUCE_PATIENCE, 
        verbose=True,
        min_lr=cfg.MIN_LR
    )
    
    # MixUp probability - sẽ áp dụng mixup với xác suất này
    mixup_prob = 0.1  # Giảm xuống còn 0.1 để tăng tính ổn định khi fine-tuning
    
    # Tính toán trọng số classes để giúp mô hình xử lý mất cân bằng dữ liệu
    print("Tính toán class weights...")
    
    # Khởi tạo mảng đếm số lượng mẫu tích cực
    positive_counts = torch.zeros(num_classes)
    total_samples = len(train_dataset)
    
    # Đếm số lượng mẫu tích cực cho mỗi class
    for _, targets in tqdm(train_loader, desc="Calculating class weights"):
        positive_counts += targets.sum(dim=0)
    
    # Tính class weights - inverse frequency với smoothing
    class_weights = total_samples / (positive_counts + 10)
    
    # Normalize để tránh weights quá lớn
    class_weights = class_weights / class_weights.mean()
    
    # Giới hạn weights trong khoảng [0.5, 5.0]
    class_weights = torch.clamp(class_weights, 0.5, 5.0)
    
    print(f"Class weights range: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    
    # Create criterion - sử dụng Enhanced Focal Loss
    criterion = EnhancedFocalLoss(
        gamma_pos=1.5,  # Giảm từ 2.0 xuống 1.5
        gamma_neg=2.0,  # Giảm từ 3.0/4.0 xuống 2.0 
        alpha_pos=0.35, # Tăng từ 0.25 lên 0.35
        alpha_neg=0.5   # Giảm từ 0.6/0.75 xuống 0.5
    )
    print(f"Sử dụng Enhanced Focal Loss với gamma_pos=1.5, gamma_neg=2.0, alpha_pos=0.35, alpha_neg=0.5")
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=cfg.USE_AMP)
    
    # Giá trị ban đầu cho best_val_map
    best_val_map = 0.0
    
    # Load from checkpoint if available
    if checkpoint_found:
        try:
            # Khởi tạo các biến mặc định để tránh lỗi biến chưa được định nghĩa
            latest_epoch = 0
            best_val_map = 0.0
            patience = cfg.EARLY_STOPPING_PATIENCE

            print(f"Đang tải mô hình từ checkpoint: {latest_checkpoint}")
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, latest_checkpoint)
            
            # Sửa cách tải checkpoint để tương thích hơn
            try:
                # Tắt hẳn việc kiểm tra an toàn và tải với map_location để đảm bảo hoạt động
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=cfg.DEVICE,
                    pickle_module=torch.serialization.pickle,
                    weights_only=False
                )
                print("✅ Tải checkpoint thành công")
            except Exception as e1:
                print(f"Lỗi khi tải với cách thông thường: {e1}")
                print("Thử tải với các cài đặt khác...")
                
                try:
                    # Thử tải với các cài đặt cũ hơn
                    checkpoint = torch.load(
                        checkpoint_path, 
                        map_location=cfg.DEVICE,
                        pickle_module=torch.serialization.pickle
                    )
                    print("✅ Tải checkpoint thành công với cài đặt cũ")
                except Exception as e2:
                    print(f"Lỗi khi tải với cài đặt cũ: {e2}")
                    # Nỗ lực cuối cùng - tải không an toàn
                    print("Thử tải với _load_global_opts...")
                    
                    # Thêm AST vào globals cho pickling
                    torch.serialization._load_global_opts = {'zip_safe': False, 'zip_check_skip': True, 'pickle_protocol': 2}
                    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
                    print("✅ Tải checkpoint thành công với cài đặt không an toàn")
            
            # In ra các khóa có trong checkpoint để debug
            print(f"Các khóa trong checkpoint: {list(checkpoint.keys())}")
            
            # Kiểm tra cấu trúc của checkpoint và tải tương ứng
            if 'model' in checkpoint:
                try:
                    # Đầu tiên, thử lấy state_dict từ model trong checkpoint
                    saved_model = checkpoint['model']
                    model.load_state_dict(saved_model.state_dict())
                    print("Đã tải model state_dict từ model trong checkpoint")
                except Exception as me:
                    print(f"Lỗi khi tải state_dict từ model: {me}")
                    print("Thử cách khác: xây dựng lại model mới từ checkpoint...")
                    try:
                        # Khởi tạo model mới với cùng cấu trúc từ checkpoint
                        saved_model = checkpoint['model']
                        # Đặt tham số từ model đã lưu
                        for target_param, param in zip(model.parameters(), saved_model.parameters()):
                            target_param.data.copy_(param.data)
                        print("Đã tải model bằng cách sao chép tham số")
                    except Exception as e3:
                        print(f"Không thể tải model bằng phương pháp nào: {e3}")
                        print("Bắt đầu huấn luyện từ đầu.")
                        latest_epoch = 0
            elif 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Đã tải model_state_dict từ checkpoint")
                except Exception as e:
                    print(f"Lỗi khi tải model_state_dict: {e}")
                    print("Thử tải với strict=False...")
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        print("Đã tải một phần model_state_dict với strict=False")
                    except Exception as e2:
                        print(f"Vẫn không thể tải: {e2}")
                        print("Bắt đầu huấn luyện từ đầu.")
                        latest_epoch = 0
            else:
                print("Không tìm thấy dữ liệu model trong checkpoint")
                print("Bắt đầu huấn luyện từ đầu.")
                latest_epoch = 0
                
            # Load best mAP để tiếp tục cải thiện
            if 'val_map' in checkpoint:
                best_val_map = checkpoint['val_map']
                print(f"Best validation mAP từ checkpoint: {best_val_map:.4f}")
            
            # Thử tải optimizer state nếu có
            if 'optimizer_state_dict' in checkpoint:
                try:
                    # Sửa để xử lý trường hợp khác biệt số lượng tham số
                    optimizer_state = checkpoint['optimizer_state_dict']
                    # Kiểm tra xem state có thể tải trực tiếp
                    if all(k in optimizer.state_dict() for k in optimizer_state.keys()):
                        optimizer.load_state_dict(optimizer_state)
                        print("Đã tải optimizer state")
                    else:
                        print("Cấu trúc optimizer trong checkpoint khác với hiện tại. Sử dụng optimizer mới.")
                except Exception as e:
                    print(f"Lỗi khi tải optimizer state: {e}. Sử dụng optimizer mới.")
            
            # Tiếp tục huấn luyện từ epoch tiếp theo
            if 'epoch' in checkpoint:
                latest_epoch = checkpoint['epoch'] + 1
                print(f"Tiếp tục huấn luyện từ epoch {latest_epoch}")
            else:
                latest_epoch = 0
            
            # Cập nhật start_epoch từ latest_epoch
            start_epoch = latest_epoch
                
            # Thêm mới: Điều chỉnh sau khi tải checkpoint thành công
            print("Áp dụng các điều chỉnh mới để cải thiện mAP...")
            
            # Giảm learning rate mạnh để fine-tuning từ checkpoint đã huấn luyện tốt
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.LEARNING_RATE * 0.1
            print(f"Đã điều chỉnh learning rate xuống {optimizer.param_groups[0]['lr']}")
            
            # Khởi tạo patience trước khi sử dụng
            patience = cfg.EARLY_STOPPING_PATIENCE
            
            # Đặt lại patience của Early Stopping lớn hơn để mô hình có thời gian cải thiện
            patience = max(patience, 50)
            print(f"Đã tăng early stopping patience lên {patience} epochs")
            
        except Exception as e:
            print(f"Lỗi tổng thể khi tải checkpoint: {e}")
            print("Bắt đầu huấn luyện từ đầu.")
            latest_epoch = 0
            start_epoch = 0
    else:
        print("Không tìm thấy checkpoint hợp lệ, bắt đầu huấn luyện từ đầu.")
        latest_epoch = 0
        start_epoch = 0
    
    # Thêm biến theo dõi early stopping
    early_stopping_counter = 0
    early_stopping_best_map = best_val_map  # Sử dụng giá trị từ checkpoint nếu có
    
    # Tăng patience lên để cho mô hình có thêm thời gian cải thiện
    patience = cfg.EARLY_STOPPING_PATIENCE
    print(f"Early stopping patience: {patience} epochs")
    
    # Training với mixup - sử dụng biến start_epoch đã được khởi tạo trước đó
    for epoch in range(start_epoch, cfg.EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for i, (inputs, targets) in enumerate(pbar):
            try:
                # Move to device
                inputs = inputs.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                
                # Thêm mới: Áp dụng Label Smoothing trước MixUp
                targets = smooth_targets(targets, smoothing=0.05)
                
                # Áp dụng MixUp với xác suất mixup_prob
                if random.random() < mixup_prob:
                    try:
                        inputs, targets = mixup_data(inputs, targets, alpha=0.2)
                    except Exception as e:
                        print(f"Lỗi khi áp dụng MixUp: {e}")
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast(enabled=cfg.USE_AMP):
                    outputs = model(inputs)
                    # Sử dụng class weights với focal loss
                    loss = criterion(outputs, targets, weights=class_weights)
                
                # Gradient accumulation - Tích tụ gradient trước khi cập nhật
                loss = loss / cfg.GRAD_ACCUMULATION_STEPS
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                
                # Thực hiện bước tối ưu sau khi tích lũy đủ số gradient
                if (i + 1) % cfg.GRAD_ACCUMULATION_STEPS == 0 or (i + 1 == len(train_loader)):
                    # Gradient clipping to prevent exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Cập nhật OneCycleLR scheduler sau mỗi batch
                    scheduler.step()
                    
                    # Zero gradients
                    optimizer.zero_grad()
                
                # Update metrics - nhân loss lại để log đúng giá trị
                train_loss += loss.item() * cfg.GRAD_ACCUMULATION_STEPS
                
                # Detach và chuyển sang CPU trước khi chuyển sang numpy
                with torch.no_grad():
                    probs = torch.sigmoid(outputs).detach().cpu()
                    targets_cpu = targets.detach().cpu()
                    
                    # Đảm bảo không có NaN
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        print("WARNING: NaN/Inf trong training predictions. Thay thế bằng zeros.")
                        probs[torch.isnan(probs) | torch.isinf(probs)] = 0.0
                    
                    # Chuyển đổi an toàn sang numpy
                    train_preds.append(probs.numpy().copy())
                    train_targets.append(targets_cpu.numpy().copy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': train_loss / (i + 1),
                    'lr': optimizer.param_groups[0]['lr'],
                    'gpu_mem': f"{torch.cuda.memory_reserved() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                # In thêm thông tin có ích
                if i % 50 == 0:
                    # In activation statistics nếu cần debug sâu hơn
                    with torch.no_grad():
                        mean_pred = torch.sigmoid(outputs).mean().item()
                        std_pred = torch.sigmoid(outputs).std().item()
                        print(f"Prediction stats - Mean: {mean_pred:.4f}, Std: {std_pred:.4f}")
                
                # Check for CUDA OOM and reduce batch size if needed
                if i % 10 == 0 and torch.cuda.is_available():
                    # If VRAM is almost full, try to free cache
                    if torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory > 0.95:
                        print("VRAM usage high, clearing cache...")
                        # Chủ động giải phóng bộ nhớ
                        del inputs, outputs, loss, probs, targets_cpu
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Lỗi khi xử lý batch {i}: {e}")
                # Bỏ qua batch này và tiếp tục
                continue
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        
        # Xử lý an toàn khi tính toán training metrics
        try:
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            
            # Kiểm tra lần cuối trước khi tính toán 
            if np.isnan(train_preds).any() or np.isinf(train_preds).any():
                print("WARNING: NaN/Inf trong stacked training predictions. Thay thế bằng zeros.")
                train_preds = np.nan_to_num(train_preds, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Làm tròn targets về 0 hoặc 1 trước khi tính mAP
            # Đặc biệt quan trọng khi sử dụng Label Smoothing hoặc MixUp
            train_targets_binary = np.round(train_targets)
            train_map = average_precision_score(train_targets_binary, train_preds, average='macro')
        except Exception as e:
            print(f"Lỗi khi tính toán training metrics: {e}")
            train_map = 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                # Move to device
                inputs = inputs.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                
                # Forward pass
                with autocast(enabled=cfg.USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                
                # Detach và chuyển sang CPU trước khi chuyển sang numpy
                probs = torch.sigmoid(outputs).detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                # Đảm bảo không có NaN trước khi chuyển sang numpy
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print("WARNING: NaN/Inf trong validation predictions. Thay thế bằng zeros.")
                    probs[torch.isnan(probs) | torch.isinf(probs)] = 0.0
                
                # Chuyển đổi an toàn sang numpy
                val_preds.append(probs.numpy().copy())
                val_targets.append(targets_cpu.numpy().copy())
                
                # Free up VRAM after each batch
                if torch.cuda.is_available():
                    # Chủ động giải phóng bộ nhớ
                    del inputs, outputs, loss, probs, targets_cpu
                    if torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory > 0.9:
                        torch.cuda.empty_cache()
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        
        # Xử lý an toàn khi tính toán validation metrics
        try:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            
            # Kiểm tra lần cuối trước khi tính toán
            if np.isnan(val_preds).any() or np.isinf(val_preds).any():
                print("WARNING: NaN/Inf trong stacked validation predictions. Thay thế bằng zeros.")
                val_preds = np.nan_to_num(val_preds, nan=0.0, posinf=1.0, neginf=0.0)
            
            val_map = average_precision_score(val_targets, val_preds, average='macro')
        except Exception as e:
            print(f"Lỗi khi tính toán validation metrics: {e}")
            val_map = 0.0  # Gán giá trị mặc định
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics and timing
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train mAP: {train_map:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Cập nhật ReduceLROnPlateau scheduler
        reduce_lr.step(val_map)
        
        # Save best model với thêm thông tin
        if val_map > best_val_map:
            best_val_map = val_map
            early_stopping_counter = 0  # Reset counter
            
            # Lưu model với state_dict và thông tin chi tiết hơn
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_map': val_map,
                'best_val_map': best_val_map,  # Thêm để dễ theo dõi
                'idx_to_label': idx_to_label,
                'labels': labels,
                'class_weights': class_weights,  # Lưu class weights để tái sử dụng
                # Lưu đầy đủ thông tin cấu hình
                'model_config': {
                    'n_mels': cfg.N_MELS,
                    'n_fft': cfg.N_FFT,
                    'hop_length': cfg.HOP_LENGTH,
                    'patch_size': cfg.PATCH_SIZE,
                    'dim': cfg.DIM,
                    'depth': cfg.DEPTH,
                    'heads': cfg.HEADS,
                    'mlp_dim': cfg.MLP_DIM,
                    'num_classes': num_classes,
                    'sample_rate': cfg.SAMPLE_RATE,
                    'mixup_prob': mixup_prob  # Thêm thông tin mixup
                }
            }
            
            # Save checkpoint
            checkpoint_filename = os.path.join(cfg.CHECKPOINT_DIR, f"ast_i2s_epoch_{epoch+1}_map_{val_map:.4f}.pt")
            save_model_safely(checkpoint, checkpoint_filename, f"checkpoint with mAP: {val_map:.4f}")
            
            # Save best model
            save_model_safely(checkpoint, cfg.MODEL_SAVE_PATH, "best model")
            
            print(f"New best model! Val mAP improved from {early_stopping_best_map:.4f} to {val_map:.4f}")
            early_stopping_best_map = val_map
        else:
            # Tăng counter nếu không cải thiện
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered sau {patience} epochs không cải thiện")
                break
        
        # Always save latest checkpoint for resuming
        checkpoint = {
            'model_state_dict': model.state_dict(),  # Thay đổi để lưu state_dict thay vì toàn bộ mô hình
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_map': val_map,
            'best_val_map': best_val_map,  # Thêm để tránh bug
            'idx_to_label': idx_to_label,
            'labels': labels,
            'class_weights': class_weights,  # Lưu class weights để tái sử dụng
            # Lưu đầy đủ thông tin cấu hình
            'model_config': {
                'n_mels': cfg.N_MELS,
                'n_fft': cfg.N_FFT,
                'hop_length': cfg.HOP_LENGTH,
                'patch_size': cfg.PATCH_SIZE,
                'dim': cfg.DIM,
                'depth': cfg.DEPTH,
                'heads': cfg.HEADS,
                'mlp_dim': cfg.MLP_DIM,
                'num_classes': num_classes
            }
        }
        save_model_safely(checkpoint, os.path.join(cfg.CHECKPOINT_DIR, "i2s_latest_checkpoint.pt"), "latest checkpoint")
        
        # Clear VRAM cache
        torch.cuda.empty_cache()
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            # Move to device
            inputs = inputs.to(cfg.DEVICE)
            
            # Forward pass
            with autocast(enabled=cfg.USE_AMP):
                outputs = model(inputs)
            
            # Update metrics
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            test_targets.append(targets.numpy())
            
            # Clear VRAM
            if torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory > 0.9:
                torch.cuda.empty_cache()
    
    # Calculate test metrics
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_map = average_precision_score(test_targets, test_preds, average='macro')
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Test mAP: {test_map:.4f}")
    
    # Save final model
    final_checkpoint = {
        'model': model,  # Lưu toàn bộ model
        'idx_to_label': idx_to_label,
        'labels': labels,  # Thêm labels để dễ dàng tham chiếu
        'test_map': test_map,
        # Lưu đầy đủ thông tin cấu hình
        'model_config': {
            'n_mels': cfg.N_MELS,
            'n_fft': cfg.N_FFT,
            'hop_length': cfg.HOP_LENGTH,
            'patch_size': cfg.PATCH_SIZE,
            'dim': cfg.DIM,
            'depth': cfg.DEPTH,
            'heads': cfg.HEADS,
            'mlp_dim': cfg.MLP_DIM,
            'num_classes': num_classes
        }
    }
    
    save_model_safely(final_checkpoint, "ast_fsd50k_i2s_final.pt", "final model")
    print(f"Saved final model to ast_fsd50k_i2s_final.pt")


# Thêm class mới để xử lý âm thanh dài
class LongAudioProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_size = cfg.MAX_AUDIO_LENGTH
        self.hop_size = cfg.MAX_AUDIO_LENGTH // 2  # 50% overlap
        self.preprocessor = AudioPreprocessor(cfg)  # Thêm preprocessing cho I2S
        
    def process_audio(self, waveform):
        # Tạo bản sao của waveform đầu vào
        waveform = waveform.clone()
        
        # Xử lý tín hiệu I2S trước khi segment
        try:
            waveform = self.preprocessor.process(waveform)
        except Exception as e:
            print(f"Lỗi khi xử lý tín hiệu I2S: {e}")
        
        # Chia âm thanh dài thành các đoạn ngắn với overlap
        segments = []
        start = 0
        while start < waveform.shape[1]:
            end = min(start + self.window_size, waveform.shape[1])
            segment = waveform[:, start:end].clone()  # Tạo bản sao
            
            # Pad nếu cần
            if segment.shape[1] < self.window_size:
                segment = F.pad(segment, (0, self.window_size - segment.shape[1]))
            
            segments.append(segment)
            start += self.hop_size
        
        return segments

# Cập nhật hàm inference để xử lý âm thanh dài
def inference(audio_path, model_path):
    # Load model
    register_ast_for_loading()
    
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        
        # Lấy model trực tiếp từ checkpoint
        if 'model' in checkpoint:
            # Lấy model trực tiếp
            model = checkpoint['model']
            idx_to_label = checkpoint['idx_to_label']
        else:
            # Fallback nếu checkpoint lưu theo cách cũ
            model_config = checkpoint['model_config']
            idx_to_label = checkpoint['idx_to_label']
            
            # Tạo model mới với cấu hình trong checkpoint
            temp_cfg = Config()
            for key, value in model_config.items():
                if hasattr(temp_cfg, key):
                    setattr(temp_cfg, key, value)
            
            model = AST(temp_cfg, model_config['num_classes'])
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(cfg.DEVICE)
        model.eval()
        
        print(f"Đã tải model từ {model_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Tạo bản sao để tránh lỗi stride
        waveform = waveform.clone()
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != cfg.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE)
            waveform = resampler(waveform)
            print(f"Resampled from {sr}Hz to {cfg.SAMPLE_RATE}Hz")
        
        # Xử lý âm thanh dài với I2S preprocessing
        processor = LongAudioProcessor(cfg)
        segments = processor.process_audio(waveform)
        
        print(f"Đã chia âm thanh thành {len(segments)} đoạn")
        
        # Tích lũy predictions từ các segments
        all_predictions = []
        for segment in segments:
            # Tạo bản sao segment để tránh lỗi stride
            segment = segment.clone()
            
            try:
                # Convert to mel spectrogram
                mel_spec = T.MelSpectrogram(
                    sample_rate=cfg.SAMPLE_RATE,
                    n_fft=cfg.N_FFT,
                    hop_length=cfg.HOP_LENGTH,
                    n_mels=cfg.N_MELS,
                )(segment)
                
                # Tăng cường tần số cao nếu cần
                mel_spec = processor.preprocessor.boost_high_freq(mel_spec)
                
                # Convert to decibels
                mel = T.AmplitudeToDB()(mel_spec)
                
                # Normalize
                mel = (mel - mel.mean()) / (mel.std() + 1e-8)
                
                # Make prediction
                with torch.no_grad():
                    with autocast(enabled=cfg.USE_AMP):
                        mel = mel.unsqueeze(0).to(cfg.DEVICE)
                        output = model(mel)
                        probs = torch.sigmoid(output)[0].cpu().numpy().copy()  # Tạo bản sao numpy
                        all_predictions.append(probs)
            except Exception as e:
                print(f"Lỗi khi xử lý đoạn: {e}")
                continue
        
        if not all_predictions:
            print("Không có dự đoán nào thành công!")
            return []
        
        # Kết hợp predictions từ các segments
        combined_probs = np.mean(np.array(all_predictions), axis=0)
        
        # Lọc theo ngưỡng tin cậy
        valid_indices = np.where(combined_probs > cfg.MIN_CONFIDENCE_THRESHOLD)[0]
        if len(valid_indices) == 0:
            print("Không có dự đoán nào vượt qua ngưỡng tin cậy!")
            return []
            
        valid_probs = combined_probs[valid_indices]
        
        # Sắp xếp và lấy top predictions
        top_indices = valid_indices[np.argsort(valid_probs)[::-1][:cfg.MAX_DETECTIONS]]
        predictions = [(idx_to_label[idx], combined_probs[idx]) for idx in top_indices]
        
        print(f"Kết quả nhận dạng:")
        for label, prob in predictions:
            print(f"  {label}: {prob:.4f}")
        
        return predictions
    except Exception as e:
        print(f"Lỗi trong quá trình inference: {e}")
        return []


# Simple model loading and prediction function
def predict_audio(audio_path):
    """
    A simple function to use the model for prediction
    """
    model_path = cfg.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        model_path = "ast_fsd50k_final.pt"
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            return None
    
    predictions = inference(audio_path, model_path)
    print(f"Predictions for {audio_path}:")
    for label, prob in predictions:
        print(f"{label}: {prob:.4f}")
    
    return predictions

# Thêm hàm test-time augmentation trước hàm inference
def tta_inference(audio_path, model_path, num_augmentations=5):
    """Sử dụng test-time augmentation để cải thiện độ chính xác dự đoán"""
    # Load model
    register_ast_for_loading()
    
    try:
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
        
        # Lấy model từ checkpoint
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            # Tạo model mới
            model_config = checkpoint.get('model_config', {})
            model = AST(cfg, model_config.get('num_classes', num_classes))
            model.load_state_dict(checkpoint['model_state_dict'])
        
        idx_to_label = checkpoint.get('idx_to_label', idx_to_label)
        
        model.to(cfg.DEVICE)
        model.eval()
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Tạo bản sao để tránh lỗi stride
        waveform = waveform.clone()
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != cfg.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Xử lý âm thanh dài
        processor = LongAudioProcessor(cfg)
        segments = processor.process_audio(waveform)
        
        # Tạo augmentations cho mỗi segment
        all_predictions = []
        
        for segment in segments:
            segment_predictions = []
            
            # Tạo nhiều augmentations cho mỗi segment
            for aug_idx in range(num_augmentations):
                try:
                    # Tạo bản sao để tránh lỗi stride
                    aug_segment = segment.clone()
                    
                    if aug_idx > 0:  # Skip augmentation for the first prediction (use original)
                        # Áp dụng các augmentation nhẹ
                        if aug_idx % 4 == 1:
                            # Thêm Gaussian noise nhẹ
                            noise_level = 0.001 * (aug_idx % 3 + 1)
                            aug_segment = aug_segment + torch.randn_like(aug_segment) * noise_level
                        elif aug_idx % 4 == 2:
                            # Áp dụng high-pass filter với các tần số khác nhau
                            cutoff = 60 + (aug_idx % 3) * 20
                            aug_segment = processor.preprocessor.apply_high_pass(aug_segment, cutoff)
                        elif aug_idx % 4 == 3:
                            # Áp dụng pre-emphasis với các tham số khác nhau
                            pre_emphasis = 0.95 + (aug_idx % 3) * 0.01
                            first_sample = aug_segment[:1].clone()
                            rest = aug_segment[1:] - pre_emphasis * aug_segment[:-1]
                            aug_segment = torch.cat([first_sample, rest])
                    
                    # Convert to mel spectrogram
                    mel_spec = T.MelSpectrogram(
                        sample_rate=cfg.SAMPLE_RATE,
                        n_fft=cfg.N_FFT,
                        hop_length=cfg.HOP_LENGTH,
                        n_mels=cfg.N_MELS,
                    )(aug_segment)
                    
                    # Tăng cường tần số cao
                    mel_spec = processor.preprocessor.boost_high_freq(mel_spec)
                    
                    # Convert to decibels
                    mel = T.AmplitudeToDB()(mel_spec)
                    
                    # Normalize
                    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
                    
                    # Make prediction
                    with torch.no_grad():
                        with autocast(enabled=cfg.USE_AMP):
                            mel = mel.unsqueeze(0).to(cfg.DEVICE)
                            output = model(mel)
                            probs = torch.sigmoid(output)[0].cpu().numpy()
                            segment_predictions.append(probs)
                except Exception as e:
                    print(f"Lỗi khi xử lý augmentation {aug_idx}: {e}")
                    continue
            
            # Combine augmentation predictions for this segment
            if segment_predictions:
                avg_segment_pred = np.mean(np.array(segment_predictions), axis=0)
                all_predictions.append(avg_segment_pred)
        
        if not all_predictions:
            print("Không có dự đoán nào thành công!")
            return []
        
        # Kết hợp predictions từ các segments
        combined_probs = np.mean(np.array(all_predictions), axis=0)
        
        # Lọc theo ngưỡng tin cậy
        valid_indices = np.where(combined_probs > cfg.MIN_CONFIDENCE_THRESHOLD)[0]
        if len(valid_indices) == 0:
            print("Không có dự đoán nào vượt qua ngưỡng tin cậy!")
            return []
            
        valid_probs = combined_probs[valid_indices]
        
        # Sắp xếp và lấy top predictions
        top_indices = valid_indices[np.argsort(valid_probs)[::-1][:cfg.MAX_DETECTIONS]]
        predictions = [(idx_to_label[idx], combined_probs[idx]) for idx in top_indices]
        
        print(f"Kết quả nhận dạng với test-time augmentation:")
        for label, prob in predictions:
            print(f"  {label}: {prob:.4f}")
        
        return predictions
    except Exception as e:
        print(f"Lỗi trong quá trình inference với TTA: {e}")
        return []

# Sửa đổi hàm predict_audio để sử dụng TTA
def predict_audio(audio_path):
    """Hàm dự đoán âm thanh với test-time augmentation"""
    model_path = cfg.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        model_path = "ast_fsd50k_final.pt"
        if not os.path.exists(model_path):
            # Kiểm tra trong thư mục checkpoint
            checkpoint_files = os.listdir(cfg.CHECKPOINT_DIR)
            checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt")]
            if checkpoint_files:
                model_path = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_files[0])
            else:
                print(f"Không tìm thấy mô hình")
                return None
    
    print(f"Dự đoán với test-time augmentation...")
    predictions = tta_inference(audio_path, model_path, num_augmentations=5)
    
    # Nếu không có kết quả, thử dùng phương pháp inference thông thường
    if not predictions:
        print("Thử dùng phương pháp inference thông thường...")
        predictions = inference(audio_path, model_path)
    
    return predictions

# Thêm hàm ensemble để kết hợp kết quả dự đoán từ nhiều checkpoints
def ensemble_prediction(audio_path, checkpoint_paths=None, threshold=0.3):
    """Kết hợp dự đoán từ nhiều checkpoints để cải thiện độ chính xác"""
    # Nếu không có đường dẫn checkpoint cụ thể, tự động tìm trong thư mục checkpoint
    if checkpoint_paths is None:
        checkpoint_paths = []
        if os.path.exists(cfg.CHECKPOINT_DIR):
            # Tìm các checkpoint có map cao trong tên
            checkpoint_files = os.listdir(cfg.CHECKPOINT_DIR)
            checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt")]
            
            # Thêm checkpoint mới nhất
            if "i2s_latest_checkpoint.pt" in checkpoint_files:
                checkpoint_paths.append(os.path.join(cfg.CHECKPOINT_DIR, "i2s_latest_checkpoint.pt"))
            
            # Tìm các checkpoint có mAP cao
            map_checkpoints = [f for f in checkpoint_files if "map_" in f]
            
            # Sắp xếp theo mAP giảm dần (giả định định dạng tên file chứa map_X.XXXX)
            try:
                map_checkpoints.sort(key=lambda f: float(f.split("map_")[1].split(".pt")[0]), reverse=True)
                # Lấy tối đa 3 checkpoint tốt nhất
                for ckpt in map_checkpoints[:min(3, len(map_checkpoints))]:
                    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, ckpt)
                    if checkpoint_path not in checkpoint_paths:
                        checkpoint_paths.append(checkpoint_path)
            except:
                # Nếu không phân tích được tên file, thêm các file checkpoint đầu tiên
                for ckpt in checkpoint_files[:min(3, len(checkpoint_files))]:
                    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, ckpt)
                    if checkpoint_path not in checkpoint_paths:
                        checkpoint_paths.append(checkpoint_path)
    
    if not checkpoint_paths:
        print("Không tìm thấy checkpoint nào cho ensemble!")
        return None
    
    print(f"Dự đoán ensemble từ {len(checkpoint_paths)} checkpoints:")
    for path in checkpoint_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Dự đoán từ mỗi checkpoint
    all_predictions = []
    common_idx_to_label = None
    
    for ckpt_path in checkpoint_paths:
        try:
            # Tải checkpoint
            checkpoint = torch.load(ckpt_path, map_location=cfg.DEVICE)
            
            # Lấy idx_to_label từ checkpoint đầu tiên
            if common_idx_to_label is None and 'idx_to_label' in checkpoint:
                common_idx_to_label = checkpoint['idx_to_label']
            
            # Tải model từ checkpoint
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Tạo model mới
                model_config = checkpoint.get('model_config', {})
                model = AST(cfg, model_config.get('num_classes', num_classes))
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"Không tìm thấy model trong {os.path.basename(ckpt_path)}")
                continue
            
            # Đặt model lên device
            model.to(cfg.DEVICE)
            model.eval()
            
            # Dự đoán với TTA cho mỗi model
            print(f"Dự đoán với {os.path.basename(ckpt_path)}...")
            
            # Load audio và xử lý
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.clone()
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != cfg.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Xử lý âm thanh dài
            processor = LongAudioProcessor(cfg)
            segments = processor.process_audio(waveform)
            
            # Dự đoán cho mỗi segment
            segment_predictions = []
            for segment in segments:
                try:
                    # Xử lý segment
                    mel_spec = T.MelSpectrogram(
                        sample_rate=cfg.SAMPLE_RATE,
                        n_fft=cfg.N_FFT,
                        hop_length=cfg.HOP_LENGTH,
                        n_mels=cfg.N_MELS,
                    )(segment)
                    
                    mel_spec = processor.preprocessor.boost_high_freq(mel_spec)
                    mel = T.AmplitudeToDB()(mel_spec)
                    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
                    
                    # Dự đoán
                    with torch.no_grad():
                        with autocast(enabled=cfg.USE_AMP):
                            mel = mel.unsqueeze(0).to(cfg.DEVICE)
                            output = model(mel)
                            probs = torch.sigmoid(output)[0].cpu().numpy()
                            segment_predictions.append(probs)
                except Exception as e:
                    print(f"Lỗi khi xử lý segment: {e}")
                    continue
            
            # Kết hợp dự đoán từ các segments
            if segment_predictions:
                model_prediction = np.mean(np.array(segment_predictions), axis=0)
                all_predictions.append(model_prediction)
        except Exception as e:
            print(f"Lỗi khi tải/sử dụng checkpoint {os.path.basename(ckpt_path)}: {e}")
    
    if not all_predictions or common_idx_to_label is None:
        print("Không có dự đoán nào thành công!")
        return None
    
    # Kết hợp dự đoán từ các models
    ensemble_probs = np.mean(np.array(all_predictions), axis=0)
    
    # Áp dụng threshold
    valid_indices = np.where(ensemble_probs > threshold)[0]
    if len(valid_indices) == 0:
        print(f"Không có dự đoán nào vượt qua ngưỡng {threshold}!")
        return []
    
    # Sắp xếp và lấy top predictions
    top_indices = valid_indices[np.argsort(ensemble_probs[valid_indices])[::-1][:cfg.MAX_DETECTIONS]]
    predictions = [(common_idx_to_label[idx], ensemble_probs[idx]) for idx in top_indices]
    
    print(f"Kết quả dự đoán ensemble:")
    for label, prob in predictions:
        print(f"  {label}: {prob:.4f}")
    
    return predictions

# Cập nhật hàm predict_audio để sử dụng ensemble nếu có nhiều checkpoints
def predict_audio(audio_path):
    """Hàm dự đoán âm thanh sử dụng ensemble nếu có thể"""
    # Kiểm tra xem có nhiều checkpoints không
    checkpoint_paths = []
    if os.path.exists(cfg.CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(cfg.CHECKPOINT_DIR) if f.endswith(".pt")]
        if len(checkpoint_files) >= 2:
            print(f"Tìm thấy {len(checkpoint_files)} checkpoints, thực hiện ensemble prediction...")
            return ensemble_prediction(audio_path)
    
    # Nếu không đủ checkpoints cho ensemble, sử dụng TTA
    print("Dự đoán với test-time augmentation...")
    
    # Tìm checkpoint tốt nhất
    model_path = cfg.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        model_path = None
        if os.path.exists(cfg.CHECKPOINT_DIR):
            checkpoint_files = [f for f in os.listdir(cfg.CHECKPOINT_DIR) if f.endswith(".pt")]
            if checkpoint_files:
                # Ưu tiên checkpoint có mAP cao
                map_checkpoints = [f for f in checkpoint_files if "map_" in f]
                if map_checkpoints:
                    try:
                        map_checkpoints.sort(key=lambda f: float(f.split("map_")[1].split(".pt")[0]), reverse=True)
                        model_path = os.path.join(cfg.CHECKPOINT_DIR, map_checkpoints[0])
                    except:
                        model_path = os.path.join(cfg.CHECKPOINT_DIR, map_checkpoints[0])
                else:
                    model_path = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_files[0])
    
    if not model_path:
        print("Không tìm thấy checkpoint!")
        return None
    
    predictions = tta_inference(audio_path, model_path)
    
    # Nếu không có kết quả, thử phương pháp inference thông thường
    if not predictions:
        print("Thử dùng phương pháp inference thông thường...")
        predictions = inference(audio_path, model_path)
    
    return predictions

def main():
    """
    Hàm main đơn giản hóa chỉ để chạy việc training model
    """
    # Kiểm tra CUDA
    if torch.cuda.is_available():
        print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA không khả dụng, đang sử dụng CPU. Quá trình này sẽ chậm hơn đáng kể.")
    
    # Thiết lập logging cơ bản
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/ast_i2s_train_{timestamp}.log"
    
    # Tạo class Logger đơn giản
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    # In thông tin cơ bản
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Thời gian bắt đầu: {datetime.datetime.now()}")
    print(f"Training model cho dữ liệu I2S với sample_rate={cfg.SAMPLE_RATE}, clip_duration={cfg.CLIP_DURATION}s")
    print(f"Mô hình AST có {cfg.DIM} dimensions, {cfg.DEPTH} layers, {cfg.HEADS} heads")
    print(f"Đang tối ưu mô hình cho độ chính xác cao nhất với âm thanh I2S")
    
    # Bắt đầu training
    print("Bắt đầu quá trình training...")
    train()


if __name__ == "__main__":
    # Import các thư viện cần thiết
    import sys
    import datetime
    main()