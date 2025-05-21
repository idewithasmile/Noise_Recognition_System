from flask import Blueprint, request, jsonify
import os
import requests
from services.yamnet_service import predict_audio

audio_bp = Blueprint('audio', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
ESP32_BASE_URL = "http://esp32_recorder.local"  # mDNS name

# ---- Gọi ESP32 bắt đầu ghi âm ----
@audio_bp.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        response = requests.post(f"{ESP32_BASE_URL}/start_recording", timeout=2)
        if response.status_code == 200:
            return jsonify({"message": "ESP32 đã bắt đầu ghi âm"}), 200
        else:
            return jsonify({"error": "ESP32 phản hồi lỗi"}), 500
    except Exception as e:
        return jsonify({"error": f"Không thể kết nối ESP32: {e}"}), 500

# ---- Gọi ESP32 dừng ghi âm ----
@audio_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        response = requests.post(f"{ESP32_BASE_URL}/stop_recording", timeout=2)
        if response.status_code == 200:
            return jsonify({"message": "ESP32 đã dừng ghi âm"}), 200
        else:
            return jsonify({"error": "ESP32 phản hồi lỗi"}), 500
    except Exception as e:
        return jsonify({"error": f"Không thể kết nối ESP32: {e}"}), 500

# ---- Dự đoán âm thanh sau khi ghi xong ----
@audio_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    top_labels = predict_audio(file_path, threshold=0.8, top_n=3)

    return jsonify({
        "predictions": [
            {"label": str(label), "score": round(float(score), 4)}
            for label, score in top_labels
        ]
    })
