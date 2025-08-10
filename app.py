import queue
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

st.set_page_config(page_title="Voice Stream (WebRTC)", layout="centered")
st.title("Voice Stream (WebRTC) — ImportError fixed")

# Use RTCConfiguration instead of removed ClientSettings
RTC_CFG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},   # public STUN
        ]
    }
)

# Simple audio level meter (as a placeholder for your pipeline e.g., speech-to-text)
class LevelMeter(AudioProcessorBase):
    def __init__(self) -> None:
        self.level_q: "queue.Queue[float]" = queue.Queue()

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()  # shape: (channels, samples)
        # Convert to mono RMS
        mono = pcm.mean(axis=0).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-9))
        try:
            self.level_q.put_nowait(rms)
        except queue.Full:
            pass
        return frame

st.info(
    "🎤 Start the mic below. This sample shows live audio levels. "
    "Replace the LevelMeter with your ASR/translation pipeline."
)

webrtc_ctx = webrtc_streamer(
    key="audio-demo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CFG,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=LevelMeter,
)

level_placeholder = st.empty()
if webrtc_ctx and webrtc_ctx.audio_processor:
    while True:
        if not webrtc_ctx.state.playing:
            break
        try:
            lvl = webrtc_ctx.audio_processor.level_q.get(timeout=1.0)
            level_placeholder.metric("Audio RMS (live)", f"{lvl:.4f}")
        except queue.Empty:
            pass
else:
    st.caption("Waiting for WebRTC session…")
