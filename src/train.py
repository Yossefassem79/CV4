import os
from face_recognition_knn_classifier import knnModel


train_dir = os.path.join("../", 'augmented_data')
model_path = os.path.join("../", 'model', 'trained_knn_model.clf')


print("Starting training...")

knn_clf = knnModel(
    train_dir=train_dir,
    model_save_path=model_path,
    n_neighbors=None,
)

print("Training complete!")



