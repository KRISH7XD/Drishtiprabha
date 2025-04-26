from queue import Queue
import threading
from TTS.api import TTS
import os

class FeedbackManager:
    def __init__(self, speech_rate, min_speech_interval, min_vibration_interval):
        # Initialize Coqui TTS
        self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        self.speech_rate = speech_rate
        self.tts_queue = Queue()
        self.tts_lock = threading.Lock()
        self.min_speech_interval = min_speech_interval
        self.min_vibration_interval = min_vibration_interval

        # Start TTS worker thread
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

    def tts_worker(self):
        """Process TTS queue."""
        while True:
            message = self.tts_queue.get()
            if message is None:
                break
            with self.tts_lock:
                # Generate and save audio file
                self.tts_model.tts_to_file(
                    text=message,
                    file_path="output.wav"
                )
                # Play the audio
                os.system("aplay output.wav")
            self.tts_queue.task_done()

    def add_to_tts_queue(self, message, priority=False):
        """Add message to TTS queue."""
        if priority:
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except:
                    pass
        self.tts_queue.put(message)

    def cleanup(self):
        """Cleanup TTS."""
        self.tts_queue.put(None)
        self.tts_thread.join()
