from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {
        "origins": [
            "http://localhost:4200",
            "https://fe-ai.vercel.app",   # prod
        ],
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "supports_credentials": True
    }},
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok", model_loaded=True)

# Example endpoint Angular might be calling
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return ('', 204)  # preflight
    data = request.get_json(silent=True) or {}
    # ... do work ...
    return jsonify(prediction="ok", input=data), 200
