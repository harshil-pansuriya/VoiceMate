import streamlit as st
import requests
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import base64

# Page config
st.set_page_config(
    page_title="VoiceMate AI", 
    page_icon="🎙️", 
    layout="centered"
)

# Custom CSS
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
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    .status-ready { background-color: #d4edda; color: #155724; }
    .status-recording { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False

# Header
st.markdown('<h1 class="main-header">🎙️ VoiceMate AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your intelligent voice assistant</p>', unsafe_allow_html=True)

# Status display
if st.session_state.recording_active:
    st.markdown('<div class="status-box status-recording">🔴 Recording in progress...</div>', unsafe_allow_html=True)
elif st.session_state.audio_data:
    st.markdown('<div class="status-box status-ready">✅ Audio ready for processing</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-box status-ready">Ready to record</div>', unsafe_allow_html=True)

# Recording configuration
SAMPLE_RATE = 16000
recording_data = []

def record_audio():
    """Simple blocking audio recording"""
    duration = st.slider("Recording duration (seconds)", 1, 15, 5)
    
    if st.button("🎙️ Record Audio", type="primary", use_container_width=True):
        st.session_state.recording_active = True
        
        with st.empty():
            for i in range(duration, 0, -1):
                st.markdown(f'<div class="status-box status-recording">🔴 Recording... {i} seconds left</div>', unsafe_allow_html=True)
                if i == duration:  # Start recording on first iteration
                    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
                sd.sleep(1000)  # Wait 1 second
        
        sd.wait()  # Wait for recording to complete
        st.session_state.recording_active = False
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio.flatten(), samplerate=SAMPLE_RATE, format='WAV')
        st.session_state.audio_data = buffer.getvalue()
        
        st.success("✅ Recording completed!")
        st.rerun()

# Recording section
if not st.session_state.audio_data:
    record_audio()

# Process recorded audio
if st.session_state.audio_data:
    st.subheader("🎵 Recorded Audio")
    st.audio(st.session_state.audio_data, format="audio/wav")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Process with AI", type="primary", use_container_width=True):
            with st.spinner("AI is processing your request..."):
                try:
                    files = {"file": ("audio.wav", st.session_state.audio_data, "audio/wav")}
                    response = requests.post("http://localhost:8080/process_voice", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        response_text = response_data.get("response_text", "No response received")
                        audio_base64 = response_data.get("audio_response", "")
                        
                        st.success("🎯 AI Response:")
                        st.write(f"**Text:** {response_text}")
                        
                        if audio_base64:
                            try:
                                audio_response = base64.b64decode(audio_base64)
                                st.subheader("🔊 Audio Response:")
                                st.audio(audio_response, format="audio/wav")
                            except Exception as decode_error:
                                st.warning(f"Could not decode audio response: {decode_error}")
                    else:
                        st.error(f"❌ Server error: {response.status_code}")
                        if response.text:
                            st.error(f"Error details: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to server. Make sure backend is running on localhost:8080")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("🔄 New Recording", use_container_width=True):
            st.session_state.audio_data = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by Whisper, LangChain, and Groq*")