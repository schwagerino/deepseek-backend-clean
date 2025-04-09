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

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    output = llm(
        user_input,
        max_tokens=256,
        temperature=0.7,
        stop=["</s>"]
    )

    return jsonify({"response": output["choices"][0]["text"]})

# Paso 4: Inicia la app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
