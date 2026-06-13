# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025


import argparse

#import panel_det
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
import os

from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from ocr_classifier import OCRClassifier

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default="", help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./validation_ocr", help='Select the validation data dir')

    args = parser.parse_args()


    # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    train_dict = load_images_dict(args.train_path)
    
    # 2) Cargar datos de validación y sus etiquetas
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    val_dict = load_images_dict(args.validation_path)

    #gt_labels = ...

    # 3) Entrenar clasificador
    if args.classifier == "lda_naive_bayes":
        from lda_normal_bayes_classifier import LdaNormalBayesClassifier
        clasificador = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    elif args.classifier == "lda_knn":
        from lda_normal_knn_classifier import LdaNormalKNNClassifier
        clasificador = LdaNormalKNNClassifier(ocr_char_size=(25, 25))

    elif args.classifier == "pca_naive_bayes":
        from pca_normal_bayes_classifier import PcaNormalBayesClassifier
        clasificador = PcaNormalBayesClassifier(ocr_char_size=(25, 25))

    else:
        print(f"Clasificador '{args.classifier}' no reconocido.")
        exit(1)

    clasificador.train(train_dict)

    # 4) Ejecutar el clasificador sobre los datos de test
    predicted_labels = clasificador.predict_dict(val_dict)
    gt_labels = clasificador.get_labels_dict(val_dict)

    # 5) Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print("Accuracy = ", accuracy)

    cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels)
    plot_confusion_matrix(cm)
    plt.show()

