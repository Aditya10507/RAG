import tempfile
import time
import os
import logging
import threading
import queue
from pathlib import Path
import speech_recognition as sr

try:
    import openai_whisper as whisper
except Exception:
    import whisper
import pyttsx3

from query import get_response

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VoiceAssistant")

class STTHandler:
    """Handles Speech-to-Text using OpenAI Whisper."""
    def __init__(self, model_name="small"):
        logger.info(f"Loading Whisper model: {model_name}")
        self.model_name = model_name
        self._model = None
        # Determine if we should use FP16 (faster on GPU, slower/unsupported on CPU)
        self.use_fp16 = False # Default to False for broader compatibility/CPU speed

    @property
    def model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        try:
            # language="en" helps speed up processing by skipping language detection
            result = self.model.transcribe(audio_path, fp16=self.use_fp16, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class TTSHandler:
    """Handles Text-to-Speech using pyttsx3."""
    def __init__(self):
        self.engine = pyttsx3.init()
        self._lock = threading.Lock()

    def speak(self, text: str):
        with self._lock:
            logger.info(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()

class AudioHandler:
    """Handles microphone access and non-blocking recording."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except (OSError, AttributeError) as e:
            logger.critical(f"Could not access microphone: {e}")
            self.microphone = None

    def listen_non_blocking(self, callback):
        """
        Starts a background thread that listens for speech and executes 
        the callback with the audio data.
        """
        if not self.microphone:
            logger.error("Microphone not initialized.")
            return None

        logger.info("Listening in background...")
        # This returns a function to stop the background listener
        return self.recognizer.listen_in_background(self.microphone, callback)

class AssistantOrchestrator:
    """Modular controller linking STT, RAG, and TTS."""
    def __init__(self):
        self.stt = STTHandler()
        self.tts = TTSHandler()
        self.audio = AudioHandler()
        self.audio_queue = queue.Queue()
        self.is_running = True

    def _audio_callback(self, recognizer, audio):
        """Callback executed by the non-blocking background listener."""
        self.audio_queue.put(audio)

    def process_loop(self):
        """Main processing loop for the queue."""
        stop_listening = self.audio.listen_non_blocking(self._audio_callback)
        
        logger.info("Voice Assistant Ready. Say 'exit' to quit.")
        
        try:
            while self.is_running:
                try:
                    # Wait for audio data from the background listener
                    audio_data = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Save audio to temporary file for Whisper
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = Path(tmp_file.name)
                try:
                    tmp_file.write(audio_data.get_wav_data())
                    tmp_file.close()

                    # 1. Transcribe
                    user_text = self.stt.transcribe(str(tmp_path))
                    if not user_text:
                        continue
                    
                    logger.info(f"User said: {user_text}")

                    # 2. Check for exit command
                    if "exit" in user_text.lower():
                        self.tts.speak("Goodbye!")
                        self.is_running = False
                        break

                    # 3. Query RAG Pipeline
                    try:
                        answer = get_response(user_text)
                    except Exception as e:
                        logger.error(f"RAG Error: {e}")
                        answer = "I'm sorry, I encountered an error while searching my database."

                    # 4. Speak
                    self.tts.speak(answer)

                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if stop_listening:
                stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    assistant = AssistantOrchestrator()
    assistant.process_loop()
