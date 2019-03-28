import cv2
from face_extractor import extract_face
from eval_face import eval_face
from time import sleep


cap = cv2.VideoCapture(0)

attribute_labels = ['Happiness', 'Sadness',
                    'Surprise', 'Anger', 'Disgust', 'Fear']

while True:
    ret, frame = cap.read()

    try:
        output = extract_face(frame)

        evaluation = eval_face(output)

        for i in range(6):
            print(attribute_labels[i], ': ', evaluation[i])

        print('\n\n')
        cv2.imshow('WebCam', output)
    except Exception:
        pass

    if cv2.waitKey(1) == 27:
        break

    sleep(0.1)
