import gradio as gr
import numpy as np
import os

# حاول تستورد face_recognition (لو متوفر)
try:
    import face_recognition
    from src.helpers import load_model, enhance_image
    model = load_model("model/trained_knn_model.clf")
    USE_MODEL = True
except:
    USE_MODEL = False


def predict(image):
    try:
        img = np.array(image)

        if not USE_MODEL:
            return "⚠️ Model not available (dlib issue)"

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

    except Exception as e:
        return f"Error: {str(e)}"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Recognition App"
)


port = int(os.environ.get("PORT", 3000))

iface.launch(
    server_name="0.0.0.0",
    server_port=port
)
