import speech_recognition as sr

class VoiceInput:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def get_command(self):
        """Listen for voice command and return text."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening for command...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Command: {command}")
                return command
            except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
                return None