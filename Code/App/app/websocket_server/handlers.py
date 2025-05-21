import json
import wave
import os
import asyncio
import time
import queue
from datetime import datetime
import websockets
from flask import current_app
from api.routes.audio_routes import sse_connections, sse_lock
from core.utils import extract_timestamp_from_filename, numpy_to_builtin_type, normalize_predictions

# Biến global cho kết nối client
esp32_client = None
client_id = None

# Sử dụng queue.Queue thay vì collections.deque vì nó thread-safe
audio_queue = queue.Queue()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Bộ đệm âm thanh để dồn dữ liệu
audio_buffer = bytearray()
start_timestamp = None
last_data_time = None

# Biến lưu trữ khoảng thời gian (ms) giữa các gói tin liên tiếp
packet_intervals = []
MAX_PACKET_HISTORY = 100  # Số lượng khoảng thời gian tối đa để tính trung bình
last_packet_time = None

# Buffer nhỏ hơn để ghi thường xuyên hơn, tránh mất mát
BUFFER_SIZE = 16000 * 2 * 2  # 2 giây
MAX_IDLE_TIME = 0.5  # 500ms

# Task monitor buffer
buffer_monitor_task = None

def save_buffer_to_file(buffer_data, timestamp):
    # Lưu buffer ra file WAV
    # print(f"[DEBUG] save_buffer_to_file - Kích thước buffer: {len(buffer_data)} bytes") 

    if len(buffer_data) == 0:
        # print("[ERROR] Buffer rỗng, không có dữ liệu để ghi")
        return None
    
    # Kiểm tra dữ liệu trong buffer có phải toàn số 0 (silent) không
    all_zeros = all(b == 0 for b in buffer_data)
    if all_zeros:
        # print("[WARNING] Buffer chỉ chứa giá trị 0, có thể là dữ liệu im lặng hoặc lỗi")
        pass
        # return None  # Không ghi file nếu buffer chỉ chứa số 0
        
    # Phân tích dữ liệu buffer
    if len(buffer_data) > 20:
        # print(f"[DEBUG] Mẫu dữ liệu buffer: {buffer_data[:10].hex()} ... {buffer_data[-10:].hex()}")
        # Đếm số byte không phải 0
        # non_zero_bytes = sum(1 for b in buffer_data if b != 0)
        #print(f"[DEBUG] Tỷ lệ byte khác 0: {non_zero_bytes}/{len(buffer_data)} ({non_zero_bytes/len(buffer_data)*100:.2f}%)")
        
        pass
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')  # Thêm milli giây để có thứ tự chính xác
    filename = f"{UPLOAD_FOLDER}/audio_{timestamp_str}.wav"

    try:
        # Ghi âm thanh vào file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)
            wf.writeframes(buffer_data)
        
        # Kiểm tra kích thước file
        file_size = os.path.getsize(filename)
        print(f"[DEBUG] Đã ghi file: {filename}, kích thước: {file_size} bytes")
        
        # Kiểm tra file có dữ liệu hợp lệ không
        with wave.open(filename, 'rb') as wf:
            frames = wf.getnframes()
            # print(f"[DEBUG] Số frame trong file WAV: {frames}")
            if frames <= 0:
                # print("[ERROR] File WAV không có frame âm thanh")
                return None
                
        return filename
    except Exception as e:
        print(f"[ERROR] Lỗi khi ghi file WAV: {e}")
        return None

async def handle_client(websocket):
    """Xử lý kết nối từ client"""
    global esp32_client, client_id, audio_buffer, start_timestamp, last_data_time, last_packet_time, packet_intervals, buffer_monitor_task

    # Lấy địa chỉ IP và port của client
    ip, port = websocket.remote_address

    # Lấy ID client từ địa chỉ kết nối
    client_id = f"{ip}_{port}"

    print(f"[+] Client mới kết nối: {ip}:{port}")

    # Lưu kết nối client
    esp32_client = websocket
    
    # Khởi động task theo dõi buffer timeout
    if buffer_monitor_task is None or buffer_monitor_task.done():
        buffer_monitor_task = asyncio.create_task(monitor_buffer_timeout())

    try:
        # Gửi thông báo chào mừng
        await websocket.send(client_id)

        # Xử lý các tin nhắn
        async for message in websocket:
            
            if isinstance(message, bytes):
                message_len = len(message)
                # print(f"[DEBUG] Nhận được {message_len} bytes từ client")
                
                if message_len > 0:
                    # print(f"[DEBUG] 10 byte đầu tiên: {message[:10].hex()}")  # Kiểm tra dữ liệu thô
                    
                    # Kiểm tra dữ liệu có dữ liệu âm thanh thực sự không 
                    non_zero = sum(1 for b in message if b != 0)
                    if non_zero == 0:
                        print("[WARNING] Gói tin chỉ chứa các byte 0, có thể là lỗi dữ liệu")
                    else:
                        # print(f"[DEBUG] Tỷ lệ byte khác 0: {non_zero}/{message_len} ({non_zero/message_len*100:.2f}%)")
                        pass

                now = time.time()
                
                # Tính khoảng thời gian giữa các gói tin
                if last_packet_time is not None:
                    interval_ms = (now - last_packet_time) * 1000  # Chuyển sang milli giây
                    
                    # Thêm vào danh sách khoảng thời gian
                    packet_intervals.append(interval_ms)
                    
                    # Giữ danh sách không quá dài
                    if len(packet_intervals) > MAX_PACKET_HISTORY:
                        packet_intervals.pop(0)
                    
                    # Phân tích khoảng thời gian mỗi 20 gói
                    if len(packet_intervals) % 20 == 0:
                        # analyze_packet_timing()
                        pass
                
                last_packet_time = now
                
                # Lưu timestamp bắt đầu ghi nếu chưa có
                if not start_timestamp:
                    start_timestamp = datetime.now()
                
                # Cập nhật thời gian nhận dữ liệu cuối cùng
                last_data_time = now

                # Dồn vào buffer
                audio_buffer.extend(message)

                # Nếu buffer đủ kích thước định sẵn
                if len(audio_buffer) >= BUFFER_SIZE:
                    filename = save_buffer_to_file(audio_buffer[:BUFFER_SIZE], start_timestamp)
                    
                    if filename:
                        audio_queue.put(filename)
                        print(f"[+] Đã ghi và đẩy vào hàng đợi: {filename}")

                    # Xóa phần đã ghi và giữ lại phần chưa ghi
                    audio_buffer = audio_buffer[BUFFER_SIZE:]
                    
                    # Đặt timestamp mới cho phần còn lại
                    start_timestamp = datetime.now()

                # Phản hồi khi nhận dữ liệu
                await websocket.send("Server đã nhận dữ liệu âm thanh")
            else:
                try:
                    data = json.loads(message)
                    print(f"[WS] Nhận JSON: {data}")
                    # Xử lý JSON data nếu cần
                except json.JSONDecodeError:
                    print(f"[WS] Nhận chuỗi thường: {message}")

    except websockets.exceptions.ConnectionClosed as e: 
        print(f"[-] Kết nối bị đóng với client {client_id}")
        print(f"[-] Kết nối bị đóng: {e.code} - {e.reason}")
        
        # Lưu buffer còn lại nếu có khi kết nối đóng
        if len(audio_buffer) > 0 and start_timestamp is not None:
            filename = save_buffer_to_file(audio_buffer, start_timestamp)
            if filename:
                audio_queue.put(filename)
            
            # Xóa buffer
            audio_buffer = bytearray()
            start_timestamp = None
            
    finally:
        # Hủy task giám sát nếu cần
        if buffer_monitor_task and not buffer_monitor_task.done():
            try:
                buffer_monitor_task.cancel()
                buffer_monitor_task = None
            except Exception as e:
                print(f"[!] Lỗi khi hủy buffer monitor: {e}")
        
        # Xóa client khi ngắt kết nối
        if esp32_client == websocket:
            esp32_client = None
            print(f"[-] Client đã ngắt kết nối: {ip}:{port}")

def analyze_packet_timing():
    """Phân tích thời gian giữa các gói dữ liệu"""
    global packet_intervals
    
    if len(packet_intervals) < 5:
        return
        
    avg_interval = sum(packet_intervals) / len(packet_intervals)
    max_interval = max(packet_intervals)
    min_interval = min(packet_intervals)
    
    print(f"[Debug] Phân tích packet: Trung bình={avg_interval:.2f}ms, Min={min_interval:.2f}ms, Max={max_interval:.2f}ms")
    
    if max_interval > 100:  # Nếu có khoảng cách lớn giữa các gói
        print(f"[!] Cảnh báo: Phát hiện khoảng hở dữ liệu {max_interval:.2f}ms > 100ms")

async def monitor_buffer_timeout():
    """Task giám sát buffer để xử lý khi không nhận được dữ liệu trong thời gian dài"""
    global audio_buffer, start_timestamp, last_data_time
    
    try:
        while True:
            await asyncio.sleep(0.1)  # Kiểm tra mỗi 100ms
            
            # Nếu có dữ liệu trong buffer và đã quá lâu không nhận thêm dữ liệu
            if (len(audio_buffer) > 0 and start_timestamp is not None and 
                last_data_time is not None and time.time() - last_data_time > MAX_IDLE_TIME):
                
                print(f"[!] Timeout - lưu {len(audio_buffer)} bytes còn lại trong buffer")
                filename = save_buffer_to_file(audio_buffer, start_timestamp)
                
                if filename:
                    audio_queue.put(filename)
                    print(f"[+] Đã lưu buffer do timeout: {filename}")
                
                # Reset buffer và timestamp
                audio_buffer = bytearray()
                start_timestamp = None
    except asyncio.CancelledError:
        print("[*] Task monitor buffer đã bị hủy")
    except Exception as e:
        print(f"[!] Lỗi trong task monitor buffer: {e}")

# Hàm này kiểm tra dữ liệu âm thanh raw từ ESP32
async def check_esp32_audio_format():
    """Gửi lệnh kiểm tra định dạng âm thanh tới ESP32"""
    global esp32_client
    
    command_data = {"command": "check_audio_format"}
    command_json = json.dumps(command_data)

    if esp32_client is not None:
        try:
            await esp32_client.send(command_json)
            print(f"Đã gửi lệnh kiểm tra định dạng âm thanh")
            return True
        except Exception as e:
            print(f"[!] Lỗi khi gửi lệnh kiểm tra: {e}")
            return False
    else:
        print("Không thể kiểm tra định dạng âm thanh, WebSocket đã đóng hoặc không kết nối.")
        return False

async def send_command_to_client(command):
    """Gửi lệnh tới client qua WebSocket"""
    global esp32_client
    
    command_data = {"command": command}
    command_json = json.dumps(command_data)

    # Kiểm tra nếu kết nối WebSocket còn mở trước khi gửi lệnh
    if esp32_client is not None:
        try:
            await esp32_client.send(command_json)
            print(f"Đã gửi lệnh đến client: {command_json}")
            return True
        except Exception as e:
            print(f"[!] Lỗi khi gửi lệnh: {e}")
            return False
    else:
        print("Không thể gửi lệnh, WebSocket đã đóng hoặc không kết nối.")
        return False

def get_audio_queue():
    """Trả về queue xử lý audio"""
    return audio_queue

# Lưu file âm thanh mẫu để kiểm tra với dữ liệu từ ESP32
def save_sample(buffer_data, note=""):
    """Lưu mẫu dữ liệu để phân tích"""
    if len(buffer_data) == 0:
        return None
        
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{UPLOAD_FOLDER}/sample_{note}_{timestamp_str}.bin"
    
    with open(filename, 'wb') as f:
        f.write(buffer_data)
    
    # print(f"[DEBUG] Đã lưu mẫu dữ liệu: {filename}, {len(buffer_data)} bytes")
    return filename

async def process_audio_queue():
    print("[*] Bắt đầu xử lý audio queue")
    try:
        while True:
            if not audio_queue.empty():
                filename = audio_queue.get()
                file_path = filename

                if not os.path.exists(file_path):
                    print(f"[!] File không tồn tại: {file_path}")
                    continue

                print(f"[*] Đang xử lý file: {filename}")

                try:
                    yamnet_service = current_app.config.get('yamnet_service')
                    ast_service = current_app.config.get('ast_service')
                    sse_service = current_app.config.get('sse_service')

                    yamnet_result = yamnet_service.predict(file_path) if yamnet_service else []
                    ast_result = ast_service.predict(file_path) if ast_service else []

                    # Đơn giản hóa thông điệp gửi SSE: chỉ tên file, dict kết quả, thời gian
                    sse_message = {
                        "type": "result_complete",
                        "file": os.path.basename(file_path),
                        "time": extract_timestamp_from_filename(file_path),
                        "results": {
                            "yamnet": normalize_predictions(yamnet_result),
                            "ast": normalize_predictions(ast_result)
                        }
                    }

                    if sse_service:
                        sse_service.send_event(sse_message)

                    print(f"[+] Đã gửi dữ liệu qua SSE: {sse_message}")

                except Exception as e:
                    error_message = {
                        "type": "error",
                        "file": os.path.basename(file_path),
                        "message": str(e),
                        "time": extract_timestamp_from_filename(file_path)
                    }
                    if sse_service:
                        sse_service.send_event(error_message)
                    print(f"[!] Lỗi xử lý {filename}: {e}")

                finally:
                    if current_app.config.get("DELETE_AFTER_PROCESS", False):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"[!] Không thể xóa file: {file_path} - {e}")
            else:
                await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("[*] Xử lý audio queue bị hủy")
    except Exception as e:
        print(f"[!] Lỗi trong xử lý audio queue: {e}")



# Thêm hàm khởi tạo websocket server để sử dụng trong app.py
async def websocket_handler(websocket, path):
    """Xử lý kết nối WebSocket"""
    await handle_client(websocket)