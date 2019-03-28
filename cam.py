import cv2
from face_extractor import FaceExtractor
from time import sleep


cap = cv2.VideoCapture(0)

attribute_labels = ['Happiness', 'Sadness',
                    'Surprise', 'Anger', 'Disgust', 'Fear']

extractor = FaceExtractor((256, 256), 0.3)

while True:
    ret, frame = cap.read()

    faces = extractor.get_faces(frame)

    if faces != None:
        for face in faces:
            cv2.imshow('Face', face)

    if cv2.waitKey(1) == 27:
        break

    sleep(0.1)
