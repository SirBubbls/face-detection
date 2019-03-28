import cv2 as cv


class FaceExtractor:
    def __init__(self, size, padding=0):
        self.w = size[0]
        self.h = size[1]
        self.padding = padding

    def __crop(self, image, x, y, w, h):

        # Calculating Padding
        padding = int((((w + h) / 2) * self.padding) / 2)

        # Fitting Image to a 1:1 format
        if w < h:
            diff = h - w

            w += diff / 2
            x -= diff / 2
        elif h < w:
            diff = w - h

            h += diff / 2
            y -= diff / 2

        # Adding Padding
        y -= padding
        x -= padding

        # Returning Cropped Image
        try:
            return image[y:y + h + 2 * padding, x:x + 2 * padding + w]
        except Exception:
            return None

    def get_faces(self, image, as_gray=True):

        # Classifier used for Face Detection
        face_classifier = cv.CascadeClassifier(
            'ressources/haarcascade_frontalface_default.xml')

        # Convert to Gray
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect Faces form Gray Image
        faces = face_classifier.detectMultiScale(image_gray, 1.1, 5)

        if len(faces) == 0:
            return None

        detected_faces = []

        for (x, y, w, h) in faces:
            try:
                # Processing Image
                cropped_image = self.__crop(image, x, y, w, h)
                rescaled = cv.resize(cropped_image, (self.w, self.h))

                # Appending face to result list
                detected_faces.append(rescaled)
            except Exception:
                return None

        return detected_faces
