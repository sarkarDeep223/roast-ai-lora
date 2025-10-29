from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_PATH = "./roast-ai-lora"

print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

prompt = "u'r so ugly when u look in the mirror, your reflection ducks"

print("🔥 Prompt:", prompt)
output = pipe(
    prompt,
    max_new_tokens=100,
    temperature=0.9,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)[0]["generated_text"]

print("\n🤖 Roast:\n", output[len(prompt):].strip())
