import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer
