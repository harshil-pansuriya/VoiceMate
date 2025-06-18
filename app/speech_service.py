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
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {str(e)}")
            raise
        
        # Initialize LLM Service with memory
        self.llm_service = LLMService()
        logger.info("Enhanced Speech service with memory initialized")

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Convert audio to text using Whisper"""
        try:
            audio, sample_rate = sf.read(io.BytesIO(audio_data))
            logger.info(f"Audio loaded: sr={sample_rate}, shape={audio.shape}")

            audio = audio.astype(np.float32)

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

    def clean_text_for_tts(self, text: str) -> str:
        """Remove markdown and JSON formatting for TTS"""
        try:
            # Strip markdown symbols (e.g., **, *, -, #)
            import re
            text = re.sub(r'[\*\#\-\_]+', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove JSON-specific characters if present
            text = re.sub(r'[\{\}\[\]\:\"\,]+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Text cleaning error: {str(e)}")
            return text


    def generate_speech(self, text: str) -> bytes:
        """Convert text to speech using gTTS"""
        try:
            if not text.strip():
                return b""
            
            clean_text = self.clean_text_for_tts(text)
            if not clean_text:
                return b""
            
            tts = gTTS(text=clean_text, lang='en', slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            
            # Convert to WAV format
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
        """Main processing pipeline with conversation memory"""
        try:
            # Step 1: Speech to Text
            query = self.transcribe_audio(audio_data)
            if not query:
                error_msg = "Sorry, I couldn't understand. Please try again."
                return error_msg, self.generate_speech(error_msg)
            
            # Step 2: Process with LLM (now with memory)
            response = self.llm_service.process_query(query)
            if not response:
                error_msg = "I couldn't process your question. Please try again."
                return error_msg, self.generate_speech(error_msg)
            
            # Step 3: Text to Speech
            audio_response = self.generate_speech(response)
            if not audio_response:
                error_msg = "Failed to generate speech for the response."
                return response, self.generate_speech(error_msg)
            
            logger.info("Voice query with memory processed successfully")
            return response, audio_response
            
        except Exception as e:
            logger.error(f"Voice processing error: {str(e)}")
            error_msg = "Technical error occurred. Please try again."
            return error_msg, self.generate_speech(error_msg)
    
    def clear_conversation_memory(self):
        """Clear conversation history"""
        self.llm_service.clear_memory()
        logger.info("Conversation memory cleared via SpeechService")