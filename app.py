from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

model = YOLO("best.pt")

def read_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(image)

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_base64 = data['image']
        img_array = read_base64_image(image_base64)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # تشغيل الموديل ورسم البوكسات
        results = model.predict(img_bgr, conf=0.5)
        result_img = results[0].plot()

        # تحويل الصورة الناتجة إلى base64
        output_base64 = image_to_base64(result_img)

        return jsonify({"detected_image": output_base64})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
