import os
import json
import tempfile
import whisper
import openai
import streamlit as st
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit config
st.set_page_config(page_title="Zoom/Live Action Item Summarizer", layout="centered")
st.title("🎙️ Zoom or Live → Action Items Extractor")

st.markdown("""
Upload your **Zoom MP4** or **record live** using mic. This app will:
1. Transcribe with Whisper
2. Summarize with GPT into action items
3. Give you a downloadable JSON
""")

# Whisper model loader
@st.cache_resource
def load_model():
    return whisper.load_model("base")

# Transcribe audio
def transcribe_audio(file_path):
    model = load_model()
    result = model.transcribe(file_path)
    return result["text"]

# GPT summarizer
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

# Audio input method
mode = st.radio("Select Input Mode", ["📤 Upload Zoom MP4", "🎤 Record Live Audio"])

audio_bytes = None
uploaded_file = None

if mode == "📤 Upload Zoom MP4":
    uploaded_file = st.file_uploader("Upload MP4", type=["mp4"])
elif mode == "🎤 Record Live Audio":
    from streamlit_audiorecorder import audiorecorder
    st.info("Click to record your meeting audio.")
    audio_bytes = audiorecorder("🔴 Start Recording", "⏹️ Stop Recording")

if uploaded_file or (audio_bytes and len(audio_bytes) > 0):
    with st.spinner("🔄 Processing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4" if uploaded_file else ".wav") as tmp:
            if uploaded_file:
                tmp.write(uploaded_file.read())
            else:
                tmp.write(audio_bytes)
            temp_path = tmp.name

        transcript = transcribe_audio(temp_path)
        st.success("✅ Transcription complete!")
        st.text_area("📝 Transcript Preview", transcript[:1000] + ("..." if len(transcript) > 1000 else ""), height=200)

        action_items_raw = summarize_to_action_items(transcript)
        try:
            action_items = json.loads(action_items_raw)
            st.subheader("📋 Action Items")
            st.json(action_items)

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(action_items, indent=4),
                file_name="action_items.json",
                mime="application/json"
            )
        except json.JSONDecodeError:
            st.error("❌ GPT returned malformed JSON.")
            st.code(action_items_raw)
