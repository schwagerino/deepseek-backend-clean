from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Paso 1: Descarga el modelo automáticamente desde Hugging Face
model_path = hf_hub_download(
    repo_id="schwagerino/deepseek-gguf",
    filename="deepseek.gguf"
)

# Paso 2: Carga el modelo
llm = Llama(
    model_path=model_path,
    n_ctx=512,       # Puedes ajustar esto según el modelo
    n_threads=4      # En Render, normalmente tienes 4 hilos
)

# Paso 3: Crea la app Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "API de DeepSeek funcionando 🚀"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Falta el prompt"}), 400

    output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
    )

    return jsonify(output)

# Paso 4: Inicia la app (solo en desarrollo local)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
