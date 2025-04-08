from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

app = Flask(__name__)

# Cargar el modelo desde la carpeta deepseek
model_name = "./deepseek"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    messages = [{"role": "user", "content": prompt}]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_tensor = input_tensor.to(model.device)

    output = model.generate(input_tensor, max_new_tokens=200)
    result = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=True)

    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
