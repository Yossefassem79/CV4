import gradio as gr
import face_recognition
import numpy as np
from src.helpers import load_model, enhance_image

model = load_model("model/trained_knn_model.clf")

def predict(image):
    img = np.array(image)
    img = enhance_image(img)

    locs = face_recognition.face_locations(img)
    if len(locs) == 0:
        return "No face detected"

    encodings = face_recognition.face_encodings(img, locs)
    closest_distances = model.kneighbors(encodings, n_neighbors=1)

    if closest_distances[0][0][0] <= 0.5:
        name = model.predict(encodings)[0]
    else:
        name = "Unknown"

    return name

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Recognition App"
)

iface.launch()
