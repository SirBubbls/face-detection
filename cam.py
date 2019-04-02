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
        cv.imshow('Face', faces[0])
    except Exception:
        pass
    # if faces != None:
        # for face in faces:
        # cv2.imshow('Face', face)

    if cv.waitKey(1) == 27:
        break

    sleep(0.1)
