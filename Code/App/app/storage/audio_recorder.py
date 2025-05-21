import os
import threading
import time
from datetime import datetime
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class AudioRecorder:
    def __init__(self, config, segment_duration=10, samplerate=16000):
        self.config = config
        self.segment_duration = segment_duration
        self.recording = False
        self.thread = None
        self.samplerate = samplerate
        self.output_dir = getattr(config, "UPLOAD_FOLDER", "uploads")

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"recording_{timestamp}.wav")

    def _record_loop(self):
        while self.recording:
            filename = self._get_filename()
            print(f"[INFO] Ghi âm vào file: {filename}")
            
            audio = sd.rec(int(self.segment_duration * self.samplerate), 
                          samplerate=self.samplerate, 
                          channels=1, dtype='int16')
            sd.wait()
            
            write(filename, self.samplerate, audio)

    def start_recording(self):
        if self.recording:
            print("[WARN] Đang ghi âm rồi.")
            return

        self.recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[INFO] Bắt đầu ghi âm liên tục.")

    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join()
        print("[INFO] Đã dừng ghi âm.")
