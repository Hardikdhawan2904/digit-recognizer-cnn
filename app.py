"""Local web UI: upload a photo of a handwritten digit, get a prediction.

Usage: python app.py, then open http://127.0.0.1:5000
"""

import base64
import io

from flask import Flask, render_template_string, request
from tensorflow import keras

from predict import preprocess

app = Flask(__name__)
model = keras.models.load_model("model.keras")  # load once at startup

PAGE = """
<!doctype html>
<title>Digit Recognizer</title>
<h1>Handwritten Digit Recognizer</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=photo accept="image/*" required>
  <button type=submit>Predict</button>
</form>
{% if result %}
  <h2>Predicted: {{ result.digit }} ({{ result.confidence }})</h2>
  <p>What the model actually saw (28x28):</p>
  <img src="data:image/png;base64,{{ result.canvas_b64 }}" width="140" height="140"
       style="image-rendering: pixelated; border:1px solid #ccc;">
{% elif error %}
  <p style="color:red">{{ error }}</p>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        photo = request.files.get("photo")
        try:
            x, canvas = preprocess(photo.stream)
            prediction = model.predict(x)
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            result = {
                "digit": int(prediction.argmax()),
                "confidence": f"{float(prediction.max()):.1%}",
                "canvas_b64": base64.b64encode(buf.getvalue()).decode(),
            }
        except Exception as e:
            error = f"Couldn't read a digit from that photo: {e}"
    return render_template_string(PAGE, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
