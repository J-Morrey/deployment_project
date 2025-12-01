import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, Request
from pydantic import BaseModel
from typing import Dict
from transformers import pipeline
import torch
import json
import os

# TTS/ASR helpers
from basic_TTS_LLM import ASRWhisper, TTSEngine, ensure_wav_mono_16k

app = FastAPI()

# ============================================================
# Load LLM
# ============================================================
llm = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)


PANTRY_FILE = "pantry.json"


# ============================================================
# Pantry Helpers
# ============================================================
def load_pantry() -> Dict:
    if not os.path.exists(PANTRY_FILE):
        with open(PANTRY_FILE, "w") as f:
            json.dump({}, f, indent=2)
        return {}
    with open(PANTRY_FILE, "r") as f:
        return json.load(f)


def save_pantry(pantry: Dict):
    with open(PANTRY_FILE, "w") as f:
        json.dump(pantry, f, indent=2)


# ============================================================
# Request Models
# ============================================================
class MealPlanRequest(BaseModel):
    prompt: str

class PantryItem(BaseModel):
    item: str
    amount: float
    unit: str

class PantryRemove(BaseModel):
    item: str
    amount: float


# ============================================================
# LLM Wrapper
# ============================================================
def call_llm(prompt: str, max_tokens: int = 350) -> str:
    """Safe wrapper for LLM."""
    try:
        out = llm(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            truncation=True
        )
        return out[0]["generated_text"]
    except Exception as e:
        raise RuntimeError(str(e))


# ============================================================
# Speech Helpers
# ============================================================
asr = None
tts = None


def get_asr():
    global asr
    if asr is None:
        asr = ASRWhisper()
    return asr


def get_tts():
    global tts
    if tts is None:
        tts = TTSEngine()
    return tts


def transcribe_upload(upload: UploadFile) -> str:
    """Persist upload, normalize, then run ASR."""
    suffix = Path(upload.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw_path = Path(tmp.name)
        tmp.write(upload.file.read())

    wav_path = raw_path.with_suffix(".wav")
    ensure_wav_mono_16k(raw_path, wav_path)
    text = get_asr().transcribe_wav(wav_path)

    # Cleanup temp files
    try:
        raw_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
    except Exception:
        pass
    return text.strip()


def synthesize_audio(text: str) -> str:
    """Run TTS and return base64-encoded wav bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = Path(tmp.name)
    get_tts().synthesize(text, out_path)
    audio_bytes = out_path.read_bytes()
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass
    return base64.b64encode(audio_bytes).decode("utf-8")


# ============================================================
# Endpoint 1: Meal Plan
# ============================================================
@app.post("/mealplan")
async def generate_meal_plan(request: Request):
    # Accept JSON (text) or multipart form with optional audio_file + prompt.
    content_type = request.headers.get("content-type", "")
    audio_file = None
    user_prompt = ""

    if "multipart/form-data" in content_type:
        form = await request.form()
        user_prompt = (form.get("prompt") or "").strip()
        audio_file = form.get("audio_file")
    elif "application/json" in content_type:
        try:
            data = await request.json()
            user_prompt = (data.get("prompt") or "").strip()
        except Exception:
            user_prompt = ""

    transcript = ""
    if audio_file is not None and hasattr(audio_file, "file"):
        transcript = transcribe_upload(audio_file)
        if transcript:
            user_prompt = user_prompt or transcript

    if not user_prompt:
        raise HTTPException(400, "Prompt cannot be empty.")

    prompt_text = f"""
You are a meal planning assistant.

Given the user's request, create:
1. A 3-7 day meal plan (breakfast/lunch/dinner)
2. A JSON shopping list

Return ONLY JSON in this form:
{{
  "meal_plan": [...],
  "shopping_list": [...]
}}

User request: "{user_prompt}"
"""

    llm_output = call_llm(prompt_text)
    response = {
        "output": llm_output,
        "input_text": user_prompt,
        "transcript": transcript
    }
    if audio_file is not None:
        response["audio_base64"] = synthesize_audio(llm_output)
        response["content_type"] = "audio/wav"
    return response

# ============================================================
# Endpoint 2: Recipe (uses pantry.json)
# ============================================================
class RecipeRequest(BaseModel):
    prompt: str

@app.post("/recipe")
async def generate_recipe(request: Request):
    """Uses pantry.json automatically. Accepts text or audio."""
    content_type = request.headers.get("content-type", "")
    audio_file = None
    user_prompt = ""

    if "multipart/form-data" in content_type:
        form = await request.form()
        user_prompt = (form.get("prompt") or "").strip()
        audio_file = form.get("audio_file")
    elif "application/json" in content_type:
        try:
            data = await request.json()
            user_prompt = (data.get("prompt") or "").strip()
        except Exception:
            user_prompt = ""

    transcript = ""
    if audio_file is not None and hasattr(audio_file, "file"):
        transcript = transcribe_upload(audio_file)
        if transcript:
            user_prompt = user_prompt or transcript

    pantry = load_pantry()

    pantry_text = "\n".join(
        f"- {item}: {info['amount']} {info['unit']}" 
        for item, info in pantry.items()
    )

    if not user_prompt:
        raise HTTPException(400, "Prompt cannot be empty.")

    prompt_text = f"""
    You are a cooking expert.

    User request:
    "{user_prompt}"

    Pantry:
    {pantry_text}

    Create a recipe that uses only pantry ingredients.
    """

    if audio_file is None:
        prompt_text +=\
        """
        Return ONLY JSON:
        {{
        "title": "",
        "ingredients": [...],
        "instructions": [...]
        }}
        """

    llm_output = call_llm(prompt_text)
    response = {
        "output": llm_output,
        "input_text": user_prompt,
        "transcript": transcript
    }
    if audio_file is not None:
        response["audio_base64"] = synthesize_audio(llm_output)
        response["content_type"] = "audio/wav"
    return response

# ============================================================
# Endpoint 3: Pantry Add
# ============================================================
@app.post("/pantry/add")
def pantry_add(item: PantryItem):
    pantry = load_pantry()
    name = item.item.lower().strip()

    # New item
    if name not in pantry:
        pantry[name] = {"amount": item.amount, "unit": item.unit}
        save_pantry(pantry)
        return {"status": "added", "item": pantry[name]}

    # Existing item — must have same units
    if pantry[name]["unit"] != item.unit:
        raise HTTPException(
            400,
            f"Unit mismatch: pantry uses '{pantry[name]['unit']}', "
            f"but request used '{item.unit}'."
        )

    pantry[name]["amount"] += item.amount
    save_pantry(pantry)
    return {"status": "updated", "item": pantry[name]}


# ============================================================
# Endpoint 4: Pantry Remove
# ============================================================
@app.post("/pantry/remove")
def pantry_remove(req: PantryRemove):
    pantry = load_pantry()
    name = req.item.lower().strip()

    # Removing item that doesn't exist = no-op
    if name not in pantry:
        return {"status": "noop", "reason": "item not in pantry"}

    # Remove amount
    pantry[name]["amount"] -= req.amount

    # If depleted, delete item
    if pantry[name]["amount"] <= 0:
        del pantry[name]
        save_pantry(pantry)
        return {"status": "removed_entirely"}

    save_pantry(pantry)
    return {"status": "updated", "name": name, "item": pantry[name]}


@app.get("/pantry.json")
def get_pantry_raw():
    return load_pantry()


# ============================================================
# Root
# ============================================================
@app.get("/")
def root():
    return {"status": "ok", "msg": "Meal Planner Backend Running"}
