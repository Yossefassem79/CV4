import gradio as gr
import numpy as np
import cv2
import os

# تحميل الصور من persons
def load_persons():
    persons = {}

    for person in os.listdir("persons"):
        person_path = os.path.join("persons", person)
        images = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            images.append(img.flatten())

        persons[person] = images

    return persons

persons = load_persons()

def predict(image):
    try:
        img = np.array(image)
        img = cv2.resize(img, (100, 100))
        img_flat = img.flatten()

        best_match = None
        best_score = float("inf")

        for person, images in persons.items():
            for ref in images:
                dist = np.linalg.norm(ref - img_flat)

                if dist < best_score:
                    best_score = dist
                    best_match = person

        if best_score < 5000:
            return f"✅ This is {best_match}"
        else:
            return "❌ Unknown"

    except Exception as e:
        return f"Error: {str(e)}"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Recognition App"
)

import os

port = int(os.environ.get("PORT", 3000))

iface.launch(
    server_name="0.0.0.0",
    server_port=port,
    show_api=False
)
