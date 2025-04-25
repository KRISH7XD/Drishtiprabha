import pyttsx3
import threading
import queue
import time

class FeedbackManager:
    def __init__(self, speech_rate=150, min_speech_interval=2.0, min_vibration_interval=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', speech_rate)
        self.tts_queue = queue.PriorityQueue()
        self.min_speech_interval = min_speech_interval
        self.min_vibration_interval = min_vibration_interval
        self.last_speech_time = 0
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def add_to_tts_queue(self, message, priority=False):
        self.tts_queue.put((0 if priority else 1, message))

    def _speech_worker(self):
        while True:
            _, message = self.tts_queue.get()
            current_time = time.time()
            if (current_time - self.last_speech_time) > self.min_speech_interval:
                self.engine.say(message)
                self.engine.runAndWait()
                self.last_speech_time = current_time
            self.tts_queue.task_done()

    def cleanup(self):
        self.engine.stop()