import os
import face_recognition

from face_recognition_knn_classifier import enhance_image
from helpers import load_model

model_path = os.path.join("../",'model', 'trained_knn_model.clf')


def predict(image_path, distance_threshold=0.5):

    knn_clf = load_model(model_path)

    image = face_recognition.load_image_file(image_path)
    image = enhance_image(image)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return []

    face_encodings = face_recognition.face_encodings(image,face_locations)

    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    # distance and index of closest neighbor for each encoding

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # are_matches = []
    # for i in range(len(face_locations)):
    #     distance = closest_distances[0][i][0]

    #     if distance <= distance_threshold:
    #         are_matches.append(True)
    #     else:
    #         are_matches.append(False)


    predictions = []

    for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches):
        name = pred if rec else "unknown"
        predictions.append((name, loc))

    return predictions


results = predict("D:/TA@BFCAI/Computer Vision/1- Khaled/4- Simple  Face Recoginition/images/test.jpg")

for name, location in results:
    print("Person:", name)