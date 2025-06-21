import streamlit as st
import requests
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import base64
import threading
from queue import Queue, Empty
import json

# Page config
st.set_page_config(page_title="VoiceMate AI", page_icon="🎙️", layout="centered")

BACKEND_URL = "https://harshil-pansuriya-voicemate-ai.hf.space/process_voice"

# Optimized CSS with reduced redundancy
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 15px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        border: 2px solid;
    }
    .status-ready { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
        color: #155724; 
        border-color: #b8dabc;
    }
    .status-recording { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
        color: #856404; 
        border-color: #f6d55c;
        animation: pulse 1.5s infinite;
    }
    .status-processing { 
        background: linear-gradient(135deg, #cce5ff 0%, #99d6ff 100%); 
        color: #004085; 
        border-color: #66c2ff;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
PLAYBACK_SPEED = 1.1

# Optimized session state initialization
@st.cache_data
def get_default_state():
    return {
        'app_state': 'ready',
        'recorded_audio': None,
        'response_data': None,
        'recording_thread': None,
        'stop_recording': False
    }

# Initialize session state efficiently
for key, value in get_default_state().items():
    if key not in st.session_state:
        st.session_state[key] = value

if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = Queue()

class OptimizedAudioRecorder:
    """Streamlined audio recorder with minimal overhead"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.audio_data = []
        
    def record(self, audio_queue, stop_flag):
        """Optimized recording method"""
        self.audio_data = []
        
        def audio_callback(indata, frames, time, status):
            if not stop_flag['stop']:
                self.audio_data.append(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=CHUNK_SIZE
            ):
                # Efficient waiting loop
                while not stop_flag['stop']:
                    sd.sleep(100)  # More efficient than time.sleep
                
                # Process audio data
                if self.audio_data:
                    audio_queue.put(np.concatenate(self.audio_data, axis=0))
                else:
                    audio_queue.put(None)
                    
        except Exception as e:
            audio_queue.put(f"Recording error: {str(e)}")

def process_audio_efficiently(audio_data):
    """Optimized audio processing"""
    try:
        # Efficient buffer creation
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, samplerate=SAMPLE_RATE, format='WAV')
        
        # Streamlined API call
        response = requests.post(
            BACKEND_URL,
            files={"file": ("audio.wav", buffer.getvalue(), "audio/wav")},
            timeout=30
        )
        
        return response.json() if response.status_code == 200 else {
            "error": f"Server error: {response.status_code}"
        }
        
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to server. Ensure backend is running on localhost:8080"}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

def play_audio_response(audio_base64):
    """Optimized audio playback"""
    try:
        audio_data, original_rate = sf.read(io.BytesIO(base64.b64decode(audio_base64)))
        sd.play(audio_data, samplerate=int(original_rate * PLAYBACK_SPEED))
        sd.wait()
    except Exception as e:
        st.error(f"Audio playback error: {str(e)}")

def cleanup_resources():
    """Efficient resource cleanup"""
    if (hasattr(st.session_state, 'recording_thread') and 
        st.session_state.recording_thread and 
        st.session_state.recording_thread.is_alive()):
        st.session_state.stop_recording = True
        st.session_state.recording_thread.join(timeout=1)
    
    st.session_state.recording_thread = None
    st.session_state.stop_recording = False

# Main UI
st.markdown('<h1 class="main-header">🎙️ VoiceMate AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your intelligent voice assistant - Click to ask!</p>', unsafe_allow_html=True)

# Centralized state management
status_placeholder = st.empty()
_, button_col, _ = st.columns([1, 1, 1])

# State-based UI rendering
if st.session_state.app_state == 'ready':
    status_placeholder.markdown(
        '<div class="status-box status-ready">✅ Ready - Ask your Question!</div>',
        unsafe_allow_html=True
    )
    
    with button_col:
        if st.button("🎙️", key="record_btn", help="Start recording", use_container_width=True):
            st.session_state.app_state = 'recording'
            st.session_state.stop_recording = False
            
            # Start recording thread
            stop_flag = {'stop': False}
            st.session_state.stop_flag = stop_flag
            
            recorder = OptimizedAudioRecorder()
            thread = threading.Thread(
                target=recorder.record,
                args=(st.session_state.audio_queue, stop_flag),
                daemon=True
            )
            thread.start()
            st.session_state.recording_thread = thread
            st.rerun()

elif st.session_state.app_state == 'recording':
    status_placeholder.markdown(
        '<div class="status-box status-recording">🔴 Recording... Click stop when finished!</div>',
        unsafe_allow_html=True
    )
    
    with button_col:
        if st.button("⏹️", key="stop_btn", help="Stop recording", use_container_width=True):
            if hasattr(st.session_state, 'stop_flag'):
                st.session_state.stop_flag['stop'] = True
            st.session_state.app_state = 'processing'
            st.rerun()
    
    # Non-blocking audio check
    try:
        audio_result = st.session_state.audio_queue.get_nowait()
        if isinstance(audio_result, str):  # Error
            st.error(f"❌ {audio_result}")
            cleanup_resources()
            st.session_state.app_state = 'ready'
            st.rerun()
        elif audio_result is not None:
            st.session_state.recorded_audio = audio_result
            st.session_state.app_state = 'processing'
            st.rerun()
    except Empty:
        pass

elif st.session_state.app_state == 'processing':
    status_placeholder.markdown(
        '<div class="status-box status-processing">🧠 Processing... Please wait</div>',
        unsafe_allow_html=True
    )
    
    # Handle audio processing
    if st.session_state.recorded_audio is None:
        try:
            audio_result = st.session_state.audio_queue.get(timeout=3)
            if isinstance(audio_result, str):
                st.error(f"❌ {audio_result}")
                cleanup_resources()
                st.session_state.app_state = 'ready'
                st.rerun()
            elif audio_result is not None:
                st.session_state.recorded_audio = audio_result
            else:
                st.error("❌ No audio recorded")
                cleanup_resources()
                st.session_state.app_state = 'ready'
                st.rerun()
        except Empty:
            st.error("❌ Recording timeout")
            cleanup_resources()
            st.session_state.app_state = 'ready'
            st.rerun()
    
    # Process audio
    if st.session_state.recorded_audio is not None:
        with st.spinner("🤖 Processing Your Question..."):
            result = process_audio_efficiently(st.session_state.recorded_audio)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.session_state.response_data = result
            
            # Reset for next interaction
            cleanup_resources()
            st.session_state.recorded_audio = None
            st.session_state.app_state = 'ready'
            st.rerun()

# Response display
if st.session_state.response_data:
    response_text = st.session_state.response_data.get("response_text", "No response")
    audio_base64 = st.session_state.response_data.get("audio_response", "")
    
    st.success("🎯 AI Response:")
    try:
        json.loads(response_text)
        st.code(response_text, language="json")
    except ValueError:
        st.markdown(response_text, unsafe_allow_html=False)
    
    if audio_base64:
        with st.spinner("🔊"):
            play_audio_response(audio_base64)
    
    st.session_state.response_data = None
    
    if st.button("🎤 Ask your next question", type="primary", use_container_width=True):
        st.rerun()

# Footer
st.markdown("---")

# Safety cleanup
if (st.session_state.app_state == 'recording' and 
    not hasattr(st.session_state, 'recording_thread')):
    st.session_state.app_state = 'ready'