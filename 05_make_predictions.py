import cv2
import numpy as np
import sqlite3
from keras.models import load_model

# Load pre-trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load the trained Keras model for sign prediction
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Define custom autocontrast function
def autocontrast(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) * (255.0 / (max_val - min_val))

# Open a connection to the SQLite database
conn = sqlite3.connect('customer_faces_data.db')
c = conn.cursor()

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face found
    for (x, y, w, h) in faces:

        # Recognize the face
        customer_uid, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])

        # Retrieve customer name from the database
        c.execute("SELECT customer_name FROM customers WHERE customer_uid LIKE ?", (f"{customer_uid}%",))
        row = c.fetchone()
        customer_name = row[0].split(" ")[0] if row else "Unknown"

        # Preprocess and make prediction on the face
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (224, 224))
        face_image = autocontrast(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = face_image.astype(np.float32) / 255.0

        # Predict sign
        prediction = model.predict(face_image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Display the sign prediction and confidence score on the frame
        cv2.putText(frame, f"Predicted Sign: {class_name}, Confidence: {confidence_score:.2f}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the recognized customer name and confidence on the frame
        if confidence > 45:
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
            cv2.rectangle(frame, (x - 22, y - 50), (x + w + 22, y - 22), (100, 180, 0), -1)
            cv2.putText(frame, f"{customer_name}: {confidence:.2f}%", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Detecting Faces and Predicting Signs...', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
