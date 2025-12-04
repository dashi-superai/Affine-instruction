#!/usr/bin/env python3
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# ----------------------------
# Load model once at startup
# ----------------------------
MODEL_PATH = "../../top_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int | None = 512
    temperature: float | None = 0.7


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Build chat prompt with template
    prompt = tokenizer.apply_chat_template(
        req.messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens or 512,
        temperature=req.temperature,
        do_sample=True,
    )

    # Decode full text
    assistant_text = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    ).strip()

    # Return OpenAI-compatible format
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": req.model,
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[-1],
            "completion_tokens": output_ids.shape[-1] - inputs.input_ids.shape[-1],
            "total_tokens": output_ids.shape[-1],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
