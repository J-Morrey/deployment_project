import os
import sys
import time
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
import soundfile as sf
from pydub import AudioSegment

# --- ASR: Whisper ---
import whisper  # uses ffmpeg under the hood

# --- LLM: Transformers ---
from transformers import pipeline as hf_pipeline

# --- TTS: Coqui TTS default (swap to VibeVoice later) ---
from TTS.api import TTS

# =========================
# Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ASR model sizes: tiny, base, small, medium, large, large-v3
ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "small")

# Larger model switch to "meta-llama/Llama-3-8b-instruct"
# LLM_MODEL_ID = os.environ.get("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")


# Coqui TTS model examples:
#   "tts_models/en/ljspeech/tacotron2-DDC"
#   "tts_models/en/vctk/vits"
#   "tts_models/en/jenny/jenny"   (fast, good quality)
TTS_MODEL_NAME = os.environ.get("TTS_MODEL", "tts_models/en/jenny/jenny")

SAMPLE_RATE = 16000  # consistent for ASR/TTS round-trip tests
TMP_DIR = Path(tempfile.gettempdir()) / "robot_voice_pipeline"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers
# =========================

def ensure_wav_mono_16k(in_path: Path, out_path: Path):
    """Convert any audio file to mono 16k WAV for Whisper."""
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    audio.export(out_path, format="wav")

def seconds_of_wav(path: Path) -> float:
    data, sr = sf.read(path)
    return len(data) / float(sr)

# =========================
# Components
# =========================

class ASRWhisper:
    def __init__(self, model_name=ASR_MODEL_NAME, device=DEVICE):
        print(f"[ASR] Loading Whisper: {model_name} on {device}")
        # whisper 'device' arg uses "cuda" or "cpu"
        self.model = whisper.load_model(model_name, device=device)

    def transcribe_wav(self, wav_path: Path) -> str:
        # Whisper likes paths or numpy arrays; we pass a wav path
        result = self.model.transcribe(str(wav_path))
        return result.get("text", "").strip()

class LLMTransformers:
    def __init__(self, model_id=LLM_MODEL_ID, device=DEVICE):
        print(f"[LLM] Loading: {model_id} on {device}")
        d = 0 if device == "cuda" else -1
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            device=d,
            torch_dtype=torch.float16 if device == "cuda" else None
        )

    def chat(self, prompt: str, max_new_tokens=64) -> str:
        # Increase tokens a bit and make sampling less strict
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=self.pipe.tokenizer.eos_token_id
        )

        # Hugging Face pipeline returns list of dicts
        txt = outputs[0].get("generated_text", "").strip()

        # Strip the original prompt if it echoes
        if txt.startswith(prompt):
            txt = txt[len(prompt):].lstrip()

        # Fallback
        if not txt:
            txt = "[No output from LLM]"

        return txt



# class LLMTransformers:
#     def __init__(self, model_id=LLM_MODEL_ID, device=DEVICE):
#         print(f"[LLM] Loading: {model_id} on {device}")
#         # device=0 selects first CUDA device when device_map is not used
#         # TinyLlama is small; for bigger models you may prefer device_map="auto"
#         d = 0 if device == "cuda" else -1
#         self.pipe = hf_pipeline(
#             "text-generation",
#             model=model_id,
#             device=d,
#             torch_dtype=torch.float16 if device == "cuda" else None
#         )

#     def chat(self, prompt: str, max_new_tokens=128) -> str:
#         out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True)
#         # pipeline returns list of dicts with 'generated_text'
#         txt = out[0]["generated_text"]
#         # For chat-style models that echo, strip the prompt when present
#         if txt.startswith(prompt):
#             txt = txt[len(prompt):].lstrip()
#         return txt.strip()

class TTSEngine:
    """
    Default: Coqui TTS. Swap to VibeVoice by replacing synthesize().

    To hook in VibeVoice later:
      - Download VibeVoice weights via Hugging Face.
      - Implement a loader/generator calling the proper VibeVoice API/CLI.
      - Keep the same synthesize(text, out_path) signature.
    """
    def __init__(self, model_name=TTS_MODEL_NAME, device=DEVICE):
        print(f"[TTS] Loading Coqui: {model_name} on {device}")
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=(device=="cuda"))

    def synthesize(self, text: str, out_path: Path, speaker=None, speed=1.0):
        self.tts.tts_to_file(text=text, file_path=str(out_path), speaker=speaker, speed=speed)
        return out_path

# =========================
# Tests
# =========================

def test_tts(tts: TTSEngine):
    print("\n[TEST] TTS")
    out = TMP_DIR / "tts_test.wav"
    tts.synthesize("Hello from the robot voice pipeline test.", out)
    dur = seconds_of_wav(out)
    assert out.exists() and dur > 0.2, f"TTS failed or too short: {out}"
    print(f"TTS OK → {out} ({dur:.2f}s)")

def test_asr(asr: ASRWhisper, tts: TTSEngine):
    print("\n[TEST] ASR")
    # Create synthetic speech then transcribe it (avoids microphone dependency)
    ref_text = "Robots are ready for deployment."
    tts_wav = TMP_DIR / "asr_source.wav"
    tts.synthesize(ref_text, tts_wav)

    # Ensure mono 16k
    mono_wav = TMP_DIR / "asr_input.wav"
    ensure_wav_mono_16k(tts_wav, mono_wav)

    hyp = asr.transcribe_wav(mono_wav)
    print(f"  Hyp: {hyp}")
    # soft check: require at least one or two key words
    ok = ("robot" in hyp.lower() or "robots" in hyp.lower()) and ("ready" in hyp.lower())
    assert ok, f"ASR did not capture expected words. Got: {hyp}"
    print(" ASR OK (captured key words)")

def test_llm(llm: LLMTransformers):
    print("\n[TEST] LLM")
    prompt = "Answer very briefly: What is 2 + 2?"
    ans = llm.chat(prompt, max_new_tokens=32).lower()
    print(f"  Ans: {ans}")
    if "4" in ans or "four" in ans:
        print("LLM OK")
    else:
        print("LLM sanity check failed (but model is running).")


def test_integration(asr: ASRWhisper, llm: LLMTransformers, tts: TTSEngine):
    print("\n[TEST] Integration (TTS → ASR → LLM → TTS)")
    # 1) TTS: create a prompt spoken aloud
    said = "Repeat the phrase: robot ready."
    entry_wav = TMP_DIR / "integration_input.wav"
    tts.synthesize(said, entry_wav)

    mono_wav = TMP_DIR / "integration_input_mono.wav"
    ensure_wav_mono_16k(entry_wav, mono_wav)

    # 2) ASR: transcribe it
    user_text = asr.transcribe_wav(mono_wav)
    print(f"  User (ASR): {user_text}")

    # 3) LLM: produce a short confirmation
    llm_prompt = f"You heard: '{user_text}'. Reply with exactly: robot ready"
    reply = llm.chat(llm_prompt, max_new_tokens=16)
    print(f"  LLM reply: {reply}")

    # 4) TTS: speak the reply
    reply_wav = TMP_DIR / "integration_reply.wav"
    tts.synthesize(reply, reply_wav)
    dur = seconds_of_wav(reply_wav)
    assert dur > 0.2, "Integration TTS output too short"
    print(f"Integration OK → {reply_wav} ({dur:.2f}s)")


def test_story(asr: ASRWhisper, llm: LLMTransformers, tts: TTSEngine):
    print("\n[TEST] Storytelling Interaction")

    # 1) Create synthetic "user" speech
    user_prompt = "Tell me a short adventure about a robot that explores Mars."
    input_wav = TMP_DIR / "story_input.wav"
    tts.synthesize(user_prompt, input_wav)

    mono_wav = TMP_DIR / "story_input_mono.wav"
    ensure_wav_mono_16k(input_wav, mono_wav)

    # 2) Transcribe (ASR)
    user_text = asr.transcribe_wav(mono_wav)
    print(f"  User said (ASR): {user_text}")

    # 3) Generate response (LLM)
    llm_prompt = f"You are a friendly robot storyteller. Reply in 3 sentences: {user_text}"
    story = llm.chat(llm_prompt, max_new_tokens=128)
    print(f"  Robot reply: {story}")

    # 4) Speak the reply
    reply_wav = TMP_DIR / "story_reply.wav"
    tts.synthesize(story, reply_wav)
    dur = seconds_of_wav(reply_wav)
    print(f"Storytelling complete → {reply_wav} ({dur:.2f}s)")


# =========================
# Main
# =========================

def main():
    print(f"Using device: {DEVICE} | tmp: {TMP_DIR}")
    asr = ASRWhisper()
    llm = LLMTransformers()
    tts = TTSEngine()

    # Run tests
    test_tts(tts)
    test_asr(asr, tts)
    test_llm(llm)
    test_integration(asr, llm, tts)
    test_story(asr, llm, tts)

    print("\nAll tests passed. You are GPU-ready.")
    print(f"WAV files stored in: {TMP_DIR}")
    print("On WSL, open a WAV from Windows with:  explorer.exe <path-to-wav>")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

