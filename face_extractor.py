import cv2 as cv
import numpy as np


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

    def blur(self, image, strength, faces):
        # Blurred Image
        tmp = image.copy()
        tmp = cv.blur(tmp, (23, 23))

        # Base Mask
        mask_shape = (image.shape[0], image.shape[1], 1)
        mask = np.full(mask_shape, 0, dtype=np.uint8)

        # For every Face Mask
        for (x, y, w, h) in faces:

            cv.ellipse(
                mask, ((int((x + x + w) / 2), int((y + y + h) / 2)), (w * 0.75, h * 1.1), 0), 255, -1)

        mask_inv = cv.bitwise_not(mask)

        img1_bg = cv.bitwise_and(image, image, mask=mask)
        img2_fg = cv.bitwise_and(tmp, tmp, mask=mask_inv)
        return cv.add(img1_bg, img2_fg)

    def get_faces(self, image, as_gray=True, mode='single'):

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
        if mode == 'blured':
            image = self.blur(image_gray, 20, faces)

        for (x, y, w, h) in faces:

            try:
                pass
                # Processing Image
                cropped_image = self.__crop(image, x, y, w, h)
                rescaled = cv.resize(cropped_image, (self.w, self.h))
                # Appending face to result list
                detected_faces.append(rescaled)
            except Exception:
                print("No Face Found")
                return None

        return detected_faces
