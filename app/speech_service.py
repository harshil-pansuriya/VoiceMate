import io
import soundfile as sf
import numpy as np
from typing import Tuple
import whisper
from gtts import gTTS
from config.logging import logger
from llm_service import LLMService

class SpeechService:
    def __init__(self):
        # Initialize Whisper for STT
        try:
            self.whisper_model = whisper.load_model("base")  # Base model is sufficient and faster
            logger.info("Whisper model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {str(e)}")
            raise
        
        # Initialize LLM Service
        self.llm_service = LLMService()
        logger.info("Speech service initialized")

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Convert audio to text using Whisper"""
        try:
            # Read audio data
            audio, sample_rate = sf.read(io.BytesIO(audio_data))
            logger.info(f"Audio loaded: sr={sample_rate}, shape={audio.shape}")

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ensure float32 format
            audio = audio.astype(np.float32)
            
            # Basic noise reduction - normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            
            # Check if audio is too quiet
            if np.max(np.abs(audio)) < 0.01:
                logger.warning("Audio too quiet")
                return ""

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio,
                fp16=False,
                language="en",
                temperature=0.0,
                best_of=1
            )
            
            transcription = result["text"].strip()
            logger.info(f"Transcription: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""

    def generate_speech(self, text: str) -> bytes:
        """Convert full text to speech using gTTS without chunking"""
        try:
            if not text.strip():
                return b""
            
            # Generate speech for the full text
            tts = gTTS(text=text.strip(), lang='en', slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            
            # Convert to standard WAV format
            audio, sr = sf.read(buffer)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio, samplerate=sr, format="WAV")
            
            logger.info(f"Generated speech for: {text[:50]}...")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Speech generation error: {str(e)}")
            return b""

    def process_voice_query(self, audio_data: bytes) -> Tuple[str, bytes]:
        """Main processing pipeline: Audio -> Text -> LLM -> Speech"""
        try:
            # Step 1: Speech to Text
            query = self.transcribe_audio(audio_data)
            if not query:
                error_msg = "Sorry, I couldn't understand. Please try again."
                return error_msg, self.generate_speech(error_msg)
            
            # Step 2: Process with LLM
            response = self.llm_service.process_query(query)
            if not response:
                error_msg = "I couldn't process your question. Please try again."
                return error_msg, self.generate_speech(error_msg)
            
            # Step 3: Text to Speech
            audio_response = self.generate_speech(response)
            if not audio_response:
                error_msg = "Failed to generate speech for the response."
                return response, self.generate_speech(error_msg)
            
            logger.info("Voice query processed successfully")
            return response, audio_response
            
        except Exception as e:
            logger.error(f"Voice processing error: {str(e)}")
            error_msg = "Technical error occurred. Please try again."
            return error_msg, self.generate_speech(error_msg)