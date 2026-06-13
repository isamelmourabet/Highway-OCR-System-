# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminar la implementación
# Modificar como se crea conveniente (incluyendo métodos y parámetros), únicamente es una guía.

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        X = []  # vectores de características, uno por imagen
        Y = []  # etiqueta numérica de cada imagen

        # Take training images and do feature extraction
        for clase, imagenes in images_dict.items():
            label = self.char2label(clase)
            for imagen in imagenes:
                feature = self._extract_feature(imagen)
                if feature is not None:
                    X.append(feature)
                    Y.append(label)

        X = np.array(X, dtype=np.float32) # Feature vectors by rows
        Y = np.array(Y, dtype=np.int32) # Labels for each row in X

        # Perform LDA training (reduce de 625 a n_clases-1 componentes)
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, Y)
        X_reduced = self.lda.transform(X)

        # Perform Classifier training
        self.classifier = GaussianNB()
        self.classifier.fit(X_reduced, Y)

        #return samples, labels

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """

        feature = self._extract_feature(img)
        if feature is None:
            return -1

        feature_reduced = self.lda.transform(feature.reshape(1, -1))
        Y = self.classifier.predict(feature_reduced) # Obtain the estimated label by the LDA + Bayes classifier
        return int(Y[0])


    def _extract_feature(self, img):
        """
        Umbraliza la imagen, recorta el contorno mayor del carácter,
        redimensiona a ocr_char_size y aplana a vector.
        """
        # umbralizar: carácter negro → blanco, fondo → negro
        thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        # quedarse con el contorno más grande (el carácter)
        contorno_mayor = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno_mayor)
        char_crop = thresh[y:y + h, x:x + w]

        # redimensionar al tamaño fijo y aplanar a vector de 625 valores
        char_resized = cv2.resize(char_crop, self.ocr_char_size)
        return char_resized.flatten().astype(np.float32)

