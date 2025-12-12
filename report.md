# Meal Planner Technical Report

## Code Repository

Code is available at https://github.com/J-Morrey/deployment_project.git

## Objectives
- Provide text/voice meal planning with a 3–7 day plan and shopping list.
- Generate pantry-aware recipes that only use available ingredients.
- Keep a lightweight pantry store for CRUD operations and allow receipt-based ingestion.
- Offer a simple Gradio UI while exposing programmatic FastAPI endpoints.

## Architecture & Data Flow
- **Frontend (frontend.py)**: Gradio Blocks UI with tabs for Pantry CRUD, Recipe Generator, Meal Plan, and Receipt Upload. Calls backend via HTTP; supports text or audio inputs and plays back synthesized speech.
- **Backend (main.py)**: FastAPI service. Core endpoints `/mealplan`, `/recipe`, `/pantry/add`, `/pantry/remove`, `/pantry.json`. Voice-capable endpoints accept multipart with `audio_file` and `prompt`. Persists pantry state in `pantry.json`.
- **Models & Runtimes**:
  - LLM: Hugging Face `pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")`, GPU if available.
  - ASR: OpenAI Whisper (`basic_TTS_LLM.ASRWhisper`) with mono 16 kHz normalization.
  - TTS: Coqui TTS (`tts_models/en/jenny/jenny`) via `basic_TTS_LLM.TTSEngine`.
  - OCR (sandbox.py): Donut `naver-clova-ix/donut-base-finetuned-cord-v2` demo for receipts (not yet wired into the FastAPI app).
- **Persistence**: `pantry.json` is a flat dict keyed by lowercased item names storing `{amount, unit}`. No DB; file locking is absent, so concurrent writes depend on single-process execution.

## Endpoint Design
- **`POST /mealplan`**: Generates a multi-day plan plus shopping list. Supports JSON body `{"prompt": "..."}` or multipart with `audio_file` and optional `prompt`. Builds a strict JSON-only prompt; returns raw LLM string in `output`, echoes `input_text`/`transcript`, and optionally `audio_base64` when audio was provided.
- **`POST /recipe`**: Pantry-aware recipe generator. Same input handling as `/mealplan`. Injects pantry contents into the prompt; asks for JSON only when text mode is used. Voice mode returns synthesized speech.
- **`POST /pantry/add`**: Adds or increments an item. Enforces unit consistency when updating. Returns status and updated item.
- **`POST /pantry/remove`**: Decrements or removes; deleting when depleted. Missing items are a no-op.
- **`GET /pantry.json`**: Raw pantry dump for UI/state refresh.
- **Missing wiring**: Receipt upload UI points to `/receipt/upload`, but that endpoint is not implemented in `main.py` yet.

## Prompting & Output Contracts
- Meal plan prompt requests JSON with `meal_plan` and `shopping_list` keys; asks for 3–7 days of meals.
- Recipe prompt injects pantry text list; constrains ingredients to the pantry. Text mode requests a JSON object with `title`, `ingredients`, and `instructions`. Voice mode allows free-form (and TTS playback).
- No response schema validation is enforced in code; callers should defensively parse.

## Model Descriptions
- **Phi-3-mini-4k-instruct (LLM)**: 4k context, lightweight; used for both meal planning and recipe generation. Sampling params: `temperature=0.7`, `max_new_tokens` default 350, truncation enabled.
- **Whisper (ASR)**: Model size selectable via `ASR_MODEL` env (default `small`). Audio normalized to mono 16 kHz WAV before transcription.
- **Coqui TTS**: `tts_models/en/jenny/jenny`, GPU-enabled when available. Outputs WAV written to a temp file then base64-encoded.
- **Donut OCR**: VisionEncoderDecoder with CORD receipt fine-tune; demo script processes `receipt.jpg` and prints parsed JSON to stdout.

## Validation & Testing
- **Voice pipeline self-tests** (`basic_TTS_LLM.py` main):
  - `test_tts`: Synthesizes fixed phrase and checks WAV duration.
  - `test_asr`: TTS → ASR loop, asserts key words present.
  - `test_llm`: Sanity math check.
  - `test_integration`: TTS → ASR → LLM → TTS round trip; asserts output duration.
  - `test_story`: End-to-end narrative generation with speech in/out (no assertions beyond duration/logging).
- **API behavior**: No automated API tests. Manual validation via `curl`/Gradio recommended:
  - Meal plan: `curl -X POST :8000/mealplan -H "Content-Type: application/json" -d '{"prompt": "4-day high-protein"}'`
  - Recipe: ensure pantry seeded, then `curl -X POST :8000/recipe ...`
  - Pantry add/remove: JSON POSTs to confirm unit enforcement and depletion handling.
- **Receipt OCR**: Only demonstrated in `sandbox.py`; not integrated or test-covered.

## Analysis & Risks
- **Robustness**: JSON contract is prompt-enforced only; malformed generations may break clients. Consider `pydantic` response validation or post-processing.
- **State consistency**: File-based pantry lacks locking; concurrent requests could overwrite. A lightweight DB (SQLite) or file lock would improve safety.
- **Security**: No auth or rate limiting. Prompt injection possible via user inputs. Audio uploads are saved to temp and deleted best-effort; no size/type validation beyond Whisper decode.
- **Performance**: GPU is preferred; CPU fallback may be slow for Phi-3/Whisper/TTS. Temp file churn for audio could be reduced with in-memory buffers.
- **Feature gap**: Receipt upload endpoint missing; OCR pipeline not connected to pantry mutation.

## Recommended Improvements
- Harden outputs with JSON parsing/repair and schema validation before returning to clients.
- Add automated endpoint tests (FastAPI TestClient) covering happy paths and error cases.
- Implement `/receipt/upload` to call Donut OCR, map items/quantities, and update `pantry.json`.
- Add pantry file locking or migrate to SQLite for concurrency safety.
- Telemetry: log prompts (with PII scrubbing), timings, and error rates for LLM/ASR/TTS calls.
- Frontend polish: add JSON syntax highlighting and clearer error toasts for failed requests.

## Deployment Notes
- Service assumes local model weights pulled via Hugging Face; offline/first-run latency can be high.
- GPU is auto-selected (`device=0`) for the LLM and TTS; Whisper uses `cuda` when available.
- Run backend: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`. Launch UI separately via `python frontend.py`.
