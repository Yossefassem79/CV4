from flask import Flask, request, jsonify
import os, face_recognition, numpy as np
from helpers import enhance_image, load_model

import sys
sys.path.insert(0, 'src')

app = Flask(__name__)
model = load_model("model/trained_knn_model.clf")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = face_recognition.load_image_file(file)
    img = enhance_image(img)
    locs = face_recognition.face_locations(img)
    if not locs:
        return jsonify({"results": []})
    encodings = face_recognition.face_encodings(img, locs)
    dists, _ = model.kneighbors(encodings, n_neighbors=1)
    results = []
    for pred, dist in zip(model.predict(encodings), dists):
        results.append({"name": pred if dist[0] <= 0.5 else "unknown"})
    return jsonify({"results": results})