from flask import Flask
import os
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})

app = Flask(__name__)

@app.get("/")
def root():
    return {"ok": True}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)