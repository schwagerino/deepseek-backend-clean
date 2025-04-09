from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Ruta de descarga del modelo
model_path = hf_hub_download(
    repo_id="schwagerino/deepseek-modelo",  # <--- nuevo repo
    filename="deepseek.gguf",               # <--- asegúrate que así se llama en Hugging Face
    repo_type="model"
)

# Carga del modelo
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0  # Puedes subir esto si Render tiene GPU
)

@app.route("/generar", methods=["POST"])
def generar():
    datos = request.json
    prompt = datos.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No se recibió ningún prompt"}), 400

    try:
        resultado = llm(prompt, max_tokens=256, stop=["</s>"])
        respuesta = resultado["choices"][0]["text"].strip()
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
