import cv2 as cv

# return cv.resize(image_data, (w, h))


class FaceExtractor:
    def __init__(self, size, padding=0):
        self.w = size[0]
        self.h = size[1]
        self.d = padding

    def get_face(self, image):
        # Classifier used for Face Detection
        face_classifier = cv.CascadeClassifier(
            './ressources/haarcascade_frontalface_default')

        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def crop(image_data, x, y, w, h):
    padding = int((((w + h) / 2) * 0.3) / 2)

    if w < h:
        diff = h - w

        w += diff / 2
        x -= diff / 2
    elif h < w:
        diff = w - h

        h += diff / 2
        y -= diff / 2

    return image_data[y - padding:y + h + padding, x - padding:x + padding + w]


def extract_face(image_data):
    face_classifier = cv.CascadeClassifier(
        './haarcascade_frontalface_default.xml')

    faces = face_classifier.detectMultiScale(image_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(image_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    try:
        cropped_image = crop(image_gray, x, y, w, h)
        cropped_image = rescale(cropped_image, 256, 256)
        # cv.imshow("cropped", cropped_image)
        return cropped_image
    except:
        return None

    try:
        print(faces.shape[0], 'faces detected')
    except:
        pass
