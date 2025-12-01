from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Load and process image
image = Image.open("receipt.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate
# Prompt tailored for receipts so the model emits a structured JSON summary.
task_prompt = (
    "<s_cord-v2>"
)
decoder_input_ids = processor.tokenizer(
    task_prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
).input_ids

outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    # Use max_new_tokens so the prompt length doesn't conflict with max_length.
    max_new_tokens=256,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Decode
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.strip()
print(processor.token2json(sequence))
