import asyncio
import threading
from flask import Blueprint, request, jsonify, current_app, copy_current_request_context, Response, stream_with_context
from websocket_server import handlers
import queue


audio_bp = Blueprint('audio', __name__)

# Biến toàn cục để theo dõi trạng thái của thread xử lý queue
processing_thread = None
processing_thread_running = False
event_loop = None

def get_or_create_event_loop():
    """Lấy event loop hiện tại hoặc tạo mới nếu cần"""
    global event_loop
    
    # Nếu đã có event loop được lưu, sử dụng nó
    if event_loop is not None and event_loop.is_running():
        return event_loop
    
    # Thử lấy event loop hiện tại, nếu không có thì tạo mới
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    event_loop = loop
    return loop

def run_async_in_event_loop(coroutine, *args, **kwargs):
    """Chạy coroutine trong event loop được quản lý"""
    loop = get_or_create_event_loop()
    
    # Nếu đang ở trong thread chính và loop đang chạy
    if threading.current_thread() is threading.main_thread() and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coroutine(*args, **kwargs), loop)
        return future
    else:
        # Nếu ở thread khác hoặc loop chưa chạy
        return loop.create_task(coroutine(*args, **kwargs))

def send_command(command):
    """Chạy lệnh gửi qua WebSocket trong thread riêng với event loop"""
    try:
        print(f"[Debug] Đang gửi lệnh: {command}")
       
        # Tạo một app context mới
        app = current_app._get_current_object()
        with app.app_context():
            loop = get_or_create_event_loop()
            if loop.is_running():
                # Nếu loop đang chạy, sử dụng run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    handlers.send_command_to_client(command), 
                    loop
                )
                future.result()  # Đợi kết quả
            else:
                # Nếu loop chưa chạy
                asyncio.run(handlers.send_command_to_client(command))
       
        print(f"[Debug] Đã gửi lệnh xong: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")

def process_audio_queue_wrapper():
    """Wrapper để chạy process_audio_queue trong event loop riêng"""
    try:
        # Tạo event loop riêng cho thread này
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Chạy coroutine process_audio_queue
        loop.run_until_complete(handlers.process_audio_queue())
    except Exception as e:
        print(f"[!] Lỗi trong quá trình xử lý audio queue: {e}")
    finally:
        try:
            # Đóng event loop một cách an toàn
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            print(f"[!] Lỗi khi đóng event loop: {e}")

def run_processing_queue(app):
    """Hàm chạy trong thread riêng để xử lý audio queue"""
    global processing_thread_running
    processing_thread_running = True
   
    print("[+] Bắt đầu thread xử lý audio queue")
    try:
        # Sử dụng app context được truyền vào
        with app.app_context():
            process_audio_queue_wrapper()
    except Exception as e:
        print(f"[!] Lỗi trong thread xử lý audio queue: {e}")
    finally:
        processing_thread_running = False
        print("[-] Thread xử lý audio queue đã kết thúc")

def ensure_processing_thread():
    """Đảm bảo thread xử lý queue đang chạy"""
    global processing_thread, processing_thread_running
   
    if processing_thread is None or not processing_thread.is_alive():
        print("[*] Khởi tạo thread xử lý audio queue mới")
        # Lấy đối tượng app hiện tại và truyền vào thread
        app = current_app._get_current_object()
        processing_thread = threading.Thread(
            target=run_processing_queue,
            args=(app,),
            daemon=True
        )
        processing_thread.start()
    else:
        print("[*] Thread xử lý audio queue đã đang chạy")


# curl.exe -X POST http://localhost:8000/api/audio/start_recording

@audio_bp.route('/start_recording', methods=['POST'])
def start_recording():
    """Bắt đầu ghi âm và gửi lệnh đến ESP32"""
    try:
        print("[API] Nhận yêu cầu bắt đầu ghi âm")
       
        ensure_processing_thread()
        # Sử dụng copy_current_request_context để bảo toàn context
        @copy_current_request_context
        def send_start_command():
            send_command("start_stream")
           
        threading.Thread(target=send_start_command).start()
        return jsonify({"status": "success", "message": "Bắt đầu ghi âm và xử lý"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# curl.exe -X POST http://localhost:8000/api/audio/stop_recording 

@audio_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Dừng ghi âm và gửi lệnh đến ESP32"""
    try:
        print("[API] Nhận yêu cầu dừng ghi âm")
       
        @copy_current_request_context
        def send_stop_command():
            send_command("stop_stream")
           
        threading.Thread(target=send_stop_command).start()
        return jsonify({"status": "success", "message": "Đã gửi lệnh dừng ghi âm"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==== SSE (Server-Sent Events) ====

sse_connections = set()
sse_lock = threading.Lock()

@audio_bp.route('/events')
def sse_events():
    q = queue.Queue()
    with sse_lock:
        sse_connections.add(q)

    def event_stream():
        try:
            while True:
                try:
                    data = q.get(timeout=0.5)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # Gửi comment giữ kết nối nếu cần (optional)
                    yield ": keep-alive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                sse_connections.discard(q)

    return Response(stream_with_context(event_stream()), content_type='text/event-stream')  

def send_sse_message(message):
    with sse_lock:
        for q in list(sse_connections):
            try:
                q.put_nowait(message)
            except queue.Full:
                sse_connections.discard(q)


def init_app(app):
    """Khởi tạo các thành phần cần thiết khi ứng dụng Flask khởi động"""
    # Không lưu event loop vào config của Flask
    # Thay vào đó, mỗi thread sẽ quản lý event loop của riêng mình
    
    # Bắt đầu thread xử lý queue
    with app.app_context():
        ensure_processing_thread()