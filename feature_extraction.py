import cv2
import numpy as np
import os

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_SIZE = 25

def char_to_label(char):
    return CHARSET.index(char)

def extract_feature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # umbralizar
    thresh = cv2.adaptiveThreshold(
        img,  # imagen en escala de grises
        255,  # valor máximo (blanco)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # cómo calcular el umbral local
        cv2.THRESH_BINARY_INV,  # invertir: carácter negro → blanco
        11,  # tamaño de la región local (debe ser impar)
        2  # constante que se resta al umbral calculado
    )

    # (necesitamos que sea caracter blanco sobre fondo negro)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None # si no hay contornos es que no ha detectado nada, por lo tanto devolvemos none

    # encontrar contorno mayor
    contorno_mayor = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    char_chop = thresh[y:y + h, x: x + w]

    # recortar y redimensionar a 25x25
    char_resize = cv2.resize(char_chop, (25, 25))

    # aplanar a vector de 625 (transformamos una especie de matriz a un vector plano)
    feature = char_resize.flatten()

    pass

def load_dataset(train_path):
    features = []
    labels = []
    # recorrer carpetas y llamar a extract_feature

    for digito in "0123456789": # recorremos la de los numeros
        carpeta = os.path.join(train_path, digito)
        label = char_to_label(digito)
        ficheros = os.listdir(carpeta)
        for fichero in ficheros:
            img_path = os.path.join(carpeta, fichero)
            feature = extract_feature(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)


    for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": # recorremos la de las letras mayusculas
        carpeta = os.path.join(train_path, "may",letra)
        label = char_to_label(letra)
        ficheros = os.listdir(carpeta)
        for fichero in ficheros:
            img_path = os.path.join(carpeta, fichero)
            feature = extract_feature(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)

    for letra in "abcdefghijklmnopqrstuvwxyz": # recorremos la de las letras minusculas
        carpeta = os.path.join(train_path, "min", letra)
        label = char_to_label(letra)
        ficheros = os.listdir(carpeta)
        for fichero in ficheros:
            img_path = os.path.join(carpeta, fichero)
            feature = extract_feature(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)