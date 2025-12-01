import base64
import json
import os
import tempfile

import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"


# ----------------------------------------------------------
# Utility: Pretty-print output
# ----------------------------------------------------------
def pretty(obj):
    try:
        return json.dumps(json.loads(obj), indent=2)
    except Exception:
        return obj


def decode_audio_to_file(b64_audio):
    if not b64_audio:
        return None
    audio_bytes = base64.b64decode(b64_audio)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


# ----------------------------------------------------------
# Pantry helpers
# ----------------------------------------------------------
def fetch_pantry():
    try:
        r = requests.get(f"{BACKEND_URL}/pantry.json")
        if r.status_code == 200:
            return pretty(json.dumps(r.json()))
    except:
        pass

    # Fallback: try loading pantry via backend root (if served there)
    try:
        r = requests.get(f"{BACKEND_URL}/recipe", timeout=1)
    except:
        pass

    # final fallback
    return "{}"


# ----------------------------------------------------------
# API Calls (text or audio)
# ----------------------------------------------------------
def send_request(endpoint, prompt="", audio_path=None):
    try:
        if audio_path:
            with open(audio_path, "rb") as f:
                files = {"audio_file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"prompt": prompt or ""}
                r = requests.post(f"{BACKEND_URL}/{endpoint}", files=files, data=data)
        else:
            r = requests.post(f"{BACKEND_URL}/{endpoint}", json={"prompt": prompt})

        if r.status_code != 200:
            return f"Error: {r.text}", None, ""

        data = r.json()
        audio_file = decode_audio_to_file(data.get("audio_base64"))
        transcript = data.get("transcript") or data.get("input_text") or prompt
        return pretty(data.get("output", "")), audio_file, transcript
    except Exception as e:
        return f"Request failed: {e}", None, ""


def call_mealplan(prompt, audio_path):
    return send_request("mealplan", prompt, audio_path)


def call_recipe(prompt, audio_path):
    return send_request("recipe", prompt, audio_path)


def handle_recipe(mode, prompt, audio_path):
    if mode == "Voice" and not audio_path:
        return "Please record or upload audio.", None, ""
    use_audio = mode == "Voice" and audio_path
    text = prompt if mode == "Text" else ""
    return call_recipe(text, audio_path if use_audio else None)


def handle_mealplan(mode, prompt, audio_path):
    if mode == "Voice" and not audio_path:
        return "Please record or upload audio.", None, ""
    use_audio = mode == "Voice" and audio_path
    text = prompt if mode == "Text" else ""
    return call_mealplan(text, audio_path if use_audio else None)


def pantry_add(item, amount, unit):
    try:
        payload = {"item": item, "amount": float(amount), "unit": unit}
        r = requests.post(f"{BACKEND_URL}/pantry/add", json=payload)
        if r.status_code != 200:
            return f"Error: {r.text}", fetch_pantry()
        return pretty(json.dumps(r.json())), fetch_pantry()
    except Exception as e:
        return f"Request failed: {e}", fetch_pantry()


def pantry_remove(item, amount):
    try:
        payload = {"item": item, "amount": float(amount)}
        r = requests.post(f"{BACKEND_URL}/pantry/remove", json=payload)
        if r.status_code != 200:
            return f"Error: {r.text}", fetch_pantry()
        return pretty(json.dumps(r.json())), fetch_pantry()
    except Exception as e:
        return f"Request failed: {e}", fetch_pantry()


# ----------------------------------------------------------
# BUILD FRONTEND
# ----------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🍽 Meal Planner & Pantry Manager")
    gr.Markdown("Use the tools below to manage pantry items and generate recipes or meal plans.")

    # ======================================================
    # Pantry Management Section
    # ======================================================
    with gr.Tab("Pantry"):
        gr.Markdown("### 🥫 Current Pantry Contents")
        pantry_display = gr.Textbox(
            label="Pantry JSON",
            value=fetch_pantry(),
            lines=12
        )

        gr.Markdown("### ➕ Add to Pantry")
        with gr.Row():
            add_item = gr.Textbox(label="Item", placeholder="e.g. rice")
            add_amount = gr.Textbox(label="Amount", placeholder="e.g. 2")
            add_unit = gr.Textbox(label="Unit", placeholder="e.g. cups")

        add_btn = gr.Button("Add to Pantry")

        add_output = gr.Textbox(label="Add Response", lines=4)

        add_btn.click(
            pantry_add,
            inputs=[add_item, add_amount, add_unit],
            outputs=[add_output, pantry_display]
        )

        gr.Markdown("### ➖ Remove From Pantry")
        with gr.Row():
            rem_item = gr.Textbox(label="Item", placeholder="e.g. rice")
            rem_amount = gr.Textbox(label="Amount", placeholder="e.g. 1")

        remove_btn = gr.Button("Remove from Pantry")
        remove_output = gr.Textbox(label="Remove Response", lines=4)

        remove_btn.click(
            pantry_remove,
            inputs=[rem_item, rem_amount],
            outputs=[remove_output, pantry_display]
        )

    # ======================================================
    # Recipe Generation Section
    # ======================================================
    with gr.Tab("Recipe Generator"):
        gr.Markdown("### 🍳 Generate Recipe Using Pantry (Text or Voice)")

        recipe_mode = gr.Radio(["Text", "Voice"], value="Text", label="Input Mode")

        recipe_prompt = gr.Textbox(
            label="Recipe Prompt",
            placeholder="e.g., Make a comforting pasta dish.",
            lines=2
        )
        recipe_audio_in = gr.Audio(
            label="Record or Upload Audio",
            sources=["microphone", "upload"],
            type="filepath"
        )

        recipe_btn = gr.Button("Generate Recipe")

        recipe_output = gr.Textbox(label="Recipe JSON", lines=12)
        recipe_audio_out = gr.Audio(label="Spoken Reply", type="filepath")
        recipe_transcript = gr.Textbox(label="Transcript / Input", lines=2)

        recipe_btn.click(
            handle_recipe,
            inputs=[recipe_mode, recipe_prompt, recipe_audio_in],
            outputs=[recipe_output, recipe_audio_out, recipe_transcript]
        )

    # ======================================================
    # Meal Plan Section
    # ======================================================
    with gr.Tab("Meal Plan"):
        gr.Markdown("### 🗓 Generate Meal Plan + Shopping List (Text or Voice)")

        mealplan_mode = gr.Radio(["Text", "Voice"], value="Text", label="Input Mode")

        mealplan_prompt = gr.Textbox(
            label="Meal Plan Prompt",
            placeholder="e.g., High-protein 4-day meal plan.",
            lines=2
        )

        mealplan_audio_in = gr.Audio(
            label="Record or Upload Audio",
            sources=["microphone", "upload"],
            type="filepath"
        )

        mealplan_btn = gr.Button("Generate Meal Plan")

        mealplan_output = gr.Textbox(label="Meal Plan JSON", lines=12)
        mealplan_audio_out = gr.Audio(label="Spoken Reply", type="filepath")
        mealplan_transcript = gr.Textbox(label="Transcript / Input", lines=2)

        mealplan_btn.click(
            handle_mealplan,
            inputs=[mealplan_mode, mealplan_prompt, mealplan_audio_in],
            outputs=[mealplan_output, mealplan_audio_out, mealplan_transcript]
        )

    with gr.Tab("Receipt Upload"):
        gr.Markdown("### 📸 Upload Receipt to Auto-Add Pantry Items")

        receipt_image = gr.Image(label="Upload Receipt", type="filepath")

        receipt_output = gr.Textbox(
            label="Receipt Parsing Output",
            lines=10
        )

        pantry_after_receipt = gr.Textbox(
            label="Updated Pantry",
            lines=10
        )

        receipt_btn = gr.Button("Process Receipt")

        def upload_receipt(image_path):
            if image_path is None:
                return "No image uploaded", "{}"

            files = {"file": open(image_path, "rb")}
            try:
                r = requests.post(f"{BACKEND_URL}/receipt/upload", files=files)
                if r.status_code != 200:
                    return f"Error: {r.text}", "{}"

                data = r.json()
                return pretty(json.dumps(data.get("items_added", []))), pretty(json.dumps(data.get("pantry_after", {})))

            except Exception as e:
                return f"Request failed: {e}", "{}"

        receipt_btn.click(
            upload_receipt,
            inputs=[receipt_image],
            outputs=[receipt_output, pantry_after_receipt]
        )

# Launch server
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
