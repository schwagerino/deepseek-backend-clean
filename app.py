from flask import Flask, request, jsonify
from llama_cpp import Llama

# Ruta completa al modelo GGUF o .bin
MODEL_PATH = "C:/Modelos/DeepSeek/deepseek.gguf"

# Cargar el modelo usando llama-cpp-python
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,  # Ajusta según tu CPU
    n_batch=8,
    verbose=True
)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    output = llm(
        prompt=prompt,
        max_tokens=256,
        stop=["</s>"],
        temperature=0.7,
        echo=False,
    )
    
    result = output["choices"][0]["text"]
    return jsonify({"response": result.strip()})

if __name__ == "__main__":
    app.run(debug=True)

