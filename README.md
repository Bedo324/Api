# 🧠 YOLO Flask API - Object Detection from Base64 Images

A lightweight Flask API for running object detection using a YOLO model (from Ultralytics) on base64-encoded images. Built to be deployed easily on platforms like Railway.

---

## 🚀 Features

- 🔍 YOLO object detection (via Ultralytics)
- 📤 Accepts base64-encoded images via POST
- 🖼️ Returns base64-encoded annotated images
- ☁️ Automatically downloads the model from Google Drive
- ⚙️ Ready for deployment on Railway (with `Procfile` and `requirements.txt`)

---

## 📁 Project Structure

- app.py # Main Flask API
- requirements.txt # Project dependencies
- Procfile # Railway process definition
- runtime.txt # Python version
- README.md # Project documentation
