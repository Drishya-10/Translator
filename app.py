import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import tempfile

from utils import (
    speech_to_text,
    translate,
    text_to_speech,
    save_audio,
    lang_map
)

st.set_page_config(page_title="AI Voice Translator")

st.title("🌍 Real-Time Voice Translator")

# Language selection
language = st.selectbox("Select Target Language", list(lang_map.keys()))

# -------------------------
# AUDIO PROCESSOR
# -------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame


# -------------------------
# START MIC
# -------------------------
webrtc_ctx = webrtc_streamer(
    key="speech",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# -------------------------
# TRANSLATE BUTTON
# -------------------------
if st.button("🎤 Translate Voice"):
    if webrtc_ctx.audio_processor:
        frames = webrtc_ctx.audio_processor.frames

        if len(frames) == 0:
            st.warning("No audio recorded")
        else:
            audio_np = np.concatenate(frames, axis=1)

            # Save audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_path = tmp.name

            save_audio(audio_np.T, temp_path)

            # Speech → Text
            text = speech_to_text(temp_path)
            st.write("📝 Recognized:", text)

            # Translation
            src, tgt, tts_code = lang_map[language]
            translated = translate(text, src, tgt)

            st.success(translated)

            # Text → Speech
            audio_file = text_to_speech(translated, tts_code)
            st.audio(audio_file)