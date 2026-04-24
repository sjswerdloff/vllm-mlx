#!/usr/bin/env python3
"""Test Qwopus with trained merger — text and vision."""

from mlx_vlm import load as vlm_load
from mlx_vlm.utils import prepare_inputs
from PIL import Image
import mlx.core as mx
import time

MODEL = "/Users/stuartswerdloff/models/Qwopus3.5-27B-v3-mxfp8-vlm-trained"
WEBCAM = "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/qwopus-3527b/webcam_snapshot.jpg"
MAX_TOKENS = 4096


def generate(model, processor, tokenizer, prompt, use_image=None):
    if use_image:
        img = Image.open(use_image).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        processed = prepare_inputs(processor, images=[img], prompts=text)
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]
        input_ids = processed["input_ids"]

        embed_output = model.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        inputs_embeds = embed_output.inputs_embeds
    else:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = mx.array([tokenizer.encode(text)])
        inputs_embeds = model.language_model.model.embed_tokens(input_ids)
        model.language_model._position_ids = None
        model.language_model._rope_deltas = None

    cache = model.language_model.make_cache()
    output = model.language_model(
        inputs=input_ids,
        inputs_embeds=inputs_embeds,
        cache=cache,
    )

    generated = []
    next_token = mx.argmax(output.logits[0, -1:, :], axis=-1)
    generated.append(next_token.item())
    mx.eval(next_token)

    for _ in range(MAX_TOKENS - 1):
        token_embed = model.language_model.model.embed_tokens(next_token.reshape(1, 1))
        output = model.language_model(
            inputs=next_token.reshape(1, 1),
            inputs_embeds=token_embed,
            cache=cache,
        )
        next_token = mx.argmax(output.logits[0, -1:, :], axis=-1)
        generated.append(next_token.item())
        mx.eval(next_token)
        if next_token.item() in (248044, 248046):
            break

    text = tokenizer.decode(generated)
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        return answer.strip(), len(generated)
    return text, len(generated)


print("Loading Qwopus with trained merger...")
t0 = time.time()
model, processor = vlm_load(MODEL)
tokenizer = processor.tokenizer
print(f"Loaded in {time.time()-t0:.1f}s\n")

# Test 1: Text only
print("=" * 60)
print("TEST 1: What is 2+2?")
print("=" * 60)
answer, n = generate(model, processor, tokenizer, "What is 2+2?")
print(f"({n} tokens)")
print(answer[:1000])

# Test 2: Text only
print("\n" + "=" * 60)
print("TEST 2: Capital of New Zealand?")
print("=" * 60)
answer, n = generate(model, processor, tokenizer, "What is the capital of New Zealand?")
print(f"({n} tokens)")
print(answer[:1000])

# Test 3: Vision
print("\n" + "=" * 60)
print("TEST 3: Webcam image")
print("=" * 60)
answer, n = generate(model, processor, tokenizer, "What do you see in this image? Describe it in detail.", use_image=WEBCAM)
print(f"({n} tokens)")
print(answer[:2000])

print("\n" + "=" * 60)
print("DONE")
