# Hướng dẫn sử dụng chương trình Training AST cho âm thanh I2S

Tài liệu này hướng dẫn cách sử dụng chương trình `training.py` để huấn luyện mô hình AST (Audio Spectrogram Transformer) cho việc nhận dạng âm thanh I2S trên bộ dữ liệu FSD50K.

## 1. Tổng quan chương trình

`training.py` là một chương trình huấn luyện mô hình AST được tối ưu hóa đặc biệt cho xử lý tín hiệu âm thanh I2S. Mô hình kết hợp kiến trúc Transformer với các kỹ thuật tiền xử lý âm thanh chuyên biệt để đạt hiệu suất cao.

Các tính năng nổi bật của chương trình:
- Tiền xử lý tín hiệu I2S chuyên biệt (high-pass filter, pre-emphasis, normalization)
- Mô hình AST với 12 lớp Transformer (768 dim, 12 heads)
- Enhanced Focal Loss với tham số tùy chỉnh cho âm thanh I2S
- Discriminative Fine-tuning
- Data augmentation đặc biệt cho âm thanh I2S
- Test-Time Augmentation và Ensemble Prediction

## 2. Cài đặt môi trường

### Yêu cầu phần cứng
- GPU với CUDA (khuyến nghị ≥ 8GB VRAM)
- RAM ≥ 16GB
- Dung lượng lưu trữ cho dataset và checkpoints

### Cài đặt thư viện
```bash
pip install torch==1.10.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas numpy matplotlib scikit-learn tqdm
```

## 3. Cấu hình bộ dữ liệu FSD50K

Trong file `training.py`, tìm và chỉnh sửa class `Config` với đường dẫn chính xác đến dữ liệu FSD50K:

```python
# Paths - Cập nhật đường dẫn thích hợp cho hệ thống của bạn
FSD50K_ROOT = "C:\\Users\\princ\\Downloads\\data\\FSD50K"  # Update with your FSD50K path
DEV_AUDIO = os.path.join(FSD50K_ROOT, "FSD50K.dev_audio")
EVAL_AUDIO = os.path.join(FSD50K_ROOT, "FSD50K.eval_audio")
GROUND_TRUTH = os.path.join(FSD50K_ROOT, "FSD50K.ground_truth")
METADATA = os.path.join(FSD50K_ROOT, "FSD50K.ground_truth")
```

## 4. Tùy chỉnh tham số mô hình

Mở file `training.py` và tùy chỉnh các tham số trong class `Config` phù hợp với nhu cầu của bạn:

### Tham số huấn luyện
```python
# Các tham số training trong code gốc
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 2e-4  # Điều chỉnh learning rate 
WEIGHT_DECAY = 2e-5   # Điều chỉnh weight decay
EPOCHS = 300          # Đủ epochs để đạt độ chính xác cao
GRAD_ACCUMULATION_STEPS = 4
EARLY_STOPPING_PATIENCE = 30
```

### Tham số xử lý I2S
Các tham số đặc biệt cho tối ưu hóa tín hiệu I2S:
```python
# Trích từ code gốc - các tham số dành riêng cho I2S
I2S_BOOST_FREQ = True        # Tăng cường tần số cao cho tín hiệu I2S
I2S_HIGH_PASS = 80           # Tần số cắt thấp (Hz) để loại bỏ nhiễu tần thấp
I2S_NORMALIZATION = True     # Chuẩn hóa biên độ tín hiệu I2S
I2S_PREEMPHASIS = 0.97       # Pre-emphasis để tăng cường tần số cao
```

### Tham số mô hình AST
```python
# Tham số kiến trúc mô hình 
SAMPLE_RATE = 16000          # Tần số lấy mẫu
N_MELS = 128                 # Số mel bands
N_FFT = 1024                 # FFT size
HOP_LENGTH = 256             # Hop length
PATCH_SIZE = 16              # Kích thước patch
DIM = 768                    # Dimension của model
DEPTH = 12                   # Số lớp transformer 
HEADS = 12                   # Số heads trong attention
MLP_DIM = 3072               # MLP dimension
```

### Tham số augmentation
```python
# Augmentation params từ code
TIME_MASK_PARAM = 60          # Time masking
FREQ_MASK_PARAM = 20          # Freq masking
TIME_STRETCH_RANGE = (0.9, 1.1) 
PITCH_SHIFT_RANGE = (-2, 2)
NOISE_SNR_RANGE = (10, 25)    # SNR range cho noise
```

### Tham số Loss function
```python
# Loss parameters từ code
FOCAL_LOSS_GAMMA = 2.0        # Thông số gamma cho focal loss
FOCAL_LOSS_ALPHA = 0.25       # Thông số alpha cho focal loss
```

## 5. Chạy quá trình huấn luyện

### Khởi động huấn luyện
```bash
python training.py
```

### Chế độ Debug
Chỉnh sửa trong Config để bật chế độ debug:
```python
DEBUG = True                   # Bật chế độ debug
DEBUG_SAMPLES = 100            # Số lượng mẫu dùng cho debug
```

## 6. Hiểu Enhanced Focal Loss

Code này sử dụng Enhanced Focal Loss đặc biệt phù hợp cho multi-label classification với âm thanh I2S:

```python
# Trích từ class EnhancedFocalLoss trong code gốc
class EnhancedFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, alpha_pos=0.25, alpha_neg=0.75, reduction='mean'):
        super(EnhancedFocalLoss, self).__init__()
        self.gamma_pos = gamma_pos  # Gamma cho positive samples
        self.gamma_neg = gamma_neg  # Gamma cao hơn cho negative samples
        self.alpha_pos = alpha_pos  # Alpha cho positive samples
        self.alpha_neg = alpha_neg  # Alpha cao hơn cho negative samples
        self.reduction = reduction
```

## 7. Hiểu Discriminative Fine-tuning

Code này áp dụng Discriminative Fine-tuning, cho phép các lớp khác nhau của mô hình học với tốc độ khác nhau:

```python
# Trích từ hàm train() trong code gốc
parameters = [
    {'params': model.patch_embed.parameters(), 'lr': cfg.LEARNING_RATE * 0.1},  # Embedding layers học chậm nhất
    {'params': transformer_params[0], 'lr': cfg.LEARNING_RATE * 0.2},           # Early layers
    {'params': transformer_params[1], 'lr': cfg.LEARNING_RATE * 0.5},           # Middle layers
    {'params': transformer_params[2], 'lr': cfg.LEARNING_RATE},                 # Late layers
    {'params': model.mlp_head.parameters(), 'lr': cfg.LEARNING_RATE * 1.2}      # Head layer học nhanh nhất
]
```

## 8. Hiểu Test-time Augmentation

Code này sử dụng Test-time Augmentation để cải thiện độ chính xác dự đoán:

```python
# Trích từ hàm tta_inference() trong code gốc
def tta_inference(audio_path, model_path, num_augmentations=5):
    """Sử dụng test-time augmentation để cải thiện độ chính xác dự đoán"""
    # ... (code implementation)
    # Tạo nhiều phiên bản biến đổi của đoạn âm thanh
    # Dự đoán trên tất cả phiên bản
    # Tổng hợp kết quả bằng trung bình
```

## 9. Giải thích Các Hàm Quan Trọng

### `TimeStretch`: Augmentation quan trọng cho âm thanh I2S
```python
# Trích từ code gốc
class TimeStretch(nn.Module):
    def __init__(self, stretch_range):
        super().__init__()
        self.stretch_range = stretch_range
        
    def forward(self, waveform):
        # ... (code implementation)
        # Thực hiện time stretching với thư viện librosa
```

### `MixUp`: Phương pháp data augmentation hiệu quả
```python
# Trích từ code gốc
def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation đặc biệt cho dữ liệu âm thanh I2S"""
    # ... (code implementation)
    # Trộn các mẫu với trọng số từ phân phối Beta
```

### `FSD50KDataset`: Định nghĩa dataset
```python
# Trích từ class FSD50KDataset trong code gốc
class FSD50KDataset(Dataset):
    def __init__(self, metadata, audio_dir, cfg, augment=False):
        # ... (initialization)
        
    def __getitem__(self, idx):
        # ... (item loading logic with I2S preprocessing)
        # Trả về mel spectrogram và target
```

## 10. Theo dõi Huấn luyện

Trong quá trình huấn luyện, các thông tin quan trọng sẽ hiển thị:
- Training loss và mAP
- Validation loss và mAP
- Learning rate hiện tại
- Thời gian huấn luyện
- Trạng thái early stopping

Mô hình tự động lưu checkpoint khi có cải thiện về mAP validation trong thư mục `checkpoints/`.

## 11. Dự đoán với Mô hình Đã Huấn luyện

Để dự đoán với mô hình đã huấn luyện:

```python
# Dự đoán bình thường
predictions = predict_audio("audio_file.wav")

# Dự đoán với TTA
predictions = tta_inference("audio_file.wav", "ast_fsd50k_i2s_model.pt")

# Dự đoán với Ensemble
predictions = ensemble_prediction("audio_file.wav")
```

## 12. Xử lý Lỗi Phổ biến

### CUDA Out of Memory
```python
# Điều chỉnh các tham số trong Config
BATCH_SIZE = 8                # Giảm batch size
EVAL_BATCH_SIZE = 16          # Giảm eval batch size
GRAD_ACCUMULATION_STEPS = 8   # Tăng gradient accumulation
```

### Quá trình huấn luyện bị treo
```python
# Giảm số workers trong Config
NUM_WORKERS = 2  # Thử với giá trị nhỏ hơn
```

### Hiệu suất mô hình kém
```python
# Điều chỉnh tham số loss function
FOCAL_LOSS_GAMMA = 1.5  # Thử với gamma thấp hơn
FOCAL_LOSS_ALPHA = 0.35 # Thử với alpha cao hơn

# Thêm bộ lọc high-pass khác
I2S_HIGH_PASS = 100  # Thử với tần số cắt cao hơn
```

