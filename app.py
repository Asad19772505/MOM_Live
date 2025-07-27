import os
import json
import tempfile
import whisper
import openai
import numpy as np
import wave
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Zoom/Live Action Item Extractor", layout="centered")
st.title("🎙️ Zoom or Live → Action Items Extractor")

st.markdown("""
Upload your **Zoom MP4** or **record live** using mic. This app will:
1. Transcribe with Whisper
2. Summarize with GPT into action items
3. Give you a downloadable JSON
""")

@st.cache_resource
def load_model():
    return whisper.load_model("base")

def transcribe_audio(file_path):
    model = load_model()
    result = model.transcribe(file_path)
    return result["text"]

def summarize_to_action_items(transcript):
    prompt = f"""
You are an expert meeting assistant. Based on the following transcript, extract all action items in structured JSON format.

Transcript:
\"\"\"{transcript}\"\"\"

Return only a JSON array like:
[
  {{
    "owner": "Name",
    "action": "What needs to be done",
    "due_date": "If mentioned, else null",
    "priority": "high/medium/low"
  }}
]
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

# Input mode selection
mode = st.radio("Select Input Mode", ["📤 Upload Zoom MP4", "🎤 Record Live Audio"])
uploaded_file = None
audio_file_path = None

# Zoom Upload Mode
if mode == "📤 Upload Zoom MP4":
    uploaded_file = st.file_uploader("Upload your Zoom MP4 file", type=["mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            audio_file_path = tmp.name

# Live Audio Recorder Mode
elif mode == "🎤 Record Live Audio":
    class AudioProcessor:
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame.to_ndarray())
            return frame

    st.info("Click Start to record your voice, then Stop and press Save & Transcribe.")
    processor = AudioProcessor()
    ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        audio_processor_factory=lambda: processor,
        client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
    )

    if st.button("🛑 Save & Transcribe"):
        if processor.frames:
            audio_data = np.concatenate(processor.frames, axis=1).flatten()
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_audio_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())
            st.success("🎧 Audio recorded!")
            audio_file_path = temp_audio_file.name
        else:
            st.warning("🎙️ No audio detected. Please record something.")

# Process if audio file exists
if audio_file_path:
    with st.spinner("Transcribing and summarizing..."):
        transcript = transcribe_audio(audio_file_path)
        st.success("✅ Transcription complete!")
        st.text_area("📝 Transcript Preview", transcript[:1000] + ("..." if len(transcript) > 1000 else ""), height=200)

        action_items_raw = summarize_to_action_items(transcript)
        try:
            action_items = json.loads(action_items_raw)
            st.subheader("📋 Action Items")
            st.json(action_items)

            st.download_button(
                label="📥 Download Action Items JSON",
                data=json.dumps(action_items, indent=4),
                file_name="action_items.json",
                mime="application/json"
            )
        except json.JSONDecodeError:
            st.error("❌ GPT returned malformed JSON.")
            st.code(action_items_raw)
