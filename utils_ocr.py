import cv2
import os

def load_images_dict(data_path):
    """
    Carga las imágenes de un directorio con estructura:
      data_path/0/ ... data_path/9/
      data_path/may/A/ ... data_path/may/Z/
      data_path/min/a/ ... data_path/min/z/
    Devuelve un diccionario {carácter: [lista de imágenes en gris]}
    """
    images_dict = {}

    # dígitos
    for digito in "0123456789":
        carpeta = os.path.join(data_path, digito)
        imagenes = []
        for fichero in sorted(os.listdir(carpeta)):
            img = cv2.imread(os.path.join(carpeta, fichero), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imagenes.append(img)
        images_dict[digito] = imagenes

    # mayúsculas
    for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        carpeta = os.path.join(data_path, "may", letra)
        imagenes = []
        for fichero in sorted(os.listdir(carpeta)):
            img = cv2.imread(os.path.join(carpeta, fichero), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imagenes.append(img)
        images_dict[letra] = imagenes

    # minúsculas
    for letra in "abcdefghijklmnopqrstuvwxyz":
        carpeta = os.path.join(data_path, "min", letra)
        imagenes = []
        for fichero in sorted(os.listdir(carpeta)):
            img = cv2.imread(os.path.join(carpeta, fichero), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imagenes.append(img)
        images_dict[letra] = imagenes

    return images_dict