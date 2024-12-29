import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Initialize some variables
known_face_encodings = []
known_face_names = []

# Load known faces (from your dataset)
def load_known_faces(images_path):
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(images_path, filename))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(img_rgb)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name

# Load known faces from the "dataset" folder (or any folder containing your training images)
load_known_faces("dataset")

# Initialize the camera (use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from BGR (OpenCV format) to RGB (Face recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize an empty list for recognized names
    face_names = []

    # Loop over all detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the detected face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name for unknown faces

        # If a match is found, use the name of the matched face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Record the attendance in a CSV file
            with open("attendance.csv", "a") as f:
                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{name},{time_now}\n")
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face with the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Press 'q' to quit the system
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
