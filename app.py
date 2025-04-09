from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import time

app = Flask(__name__)

# Paso 1: Descarga automática del modelo desde Hugging Face
print("🔄 Descargando modelo desde Hugging Face...")
model_path = hf_hub_download(
    repo_id="schwagerino/deepseek-gguf",
    filename="deepseek.gguf"
)
print("✅ Modelo descargado en:", model_path)

# Paso 2: Carga del modelo
print("🚀 Cargando modelo en memoria...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4
)
print("✅ Modelo cargado correctamente")

# Ruta base para testear si la API está viva
@app.route("/")
def home():
    return "✅ API de DeepSeek funcionando 🚀"

# Ruta principal de generación de texto
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Falta el prompt"}), 400

    print("📝 Recibido prompt:", prompt)
    start = time.time()

    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7
        )
    except Exception as e:
        print("❌ Error durante la generación:", str(e))
        return jsonify({"error": "Ocurrió un error durante la generación."}), 500

    end = time.time()
    print(f"✅ Generado en {end - start:.2f} segundos")

    return jsonify(output)

# Paso 4: Ejecutar servidor local (solo en pruebas)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
