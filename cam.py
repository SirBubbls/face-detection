import cv2 as cv
from face_extractor import FaceExtractor
from time import sleep


cap = cv.VideoCapture(0)

attribute_labels = ['Happiness', 'Sadness',
                    'Surprise', 'Anger', 'Disgust', 'Fear']

extractor = FaceExtractor((256, 256), 0.3)

while True:
    ret, frame = cap.read()

    try:
        faces = extractor.get_faces(frame, mode='blured')
        for faced_num in range(len(faces)):
            cv.imshow('Face {}'.format(faced_num), faces[faced_num])
    except Exception:
        pass
    # if faces:
        # for face in faces:
            # cv.imshow('Face', face)

    if cv.waitKey(1) == 27:
        break

    sleep(0.1)
