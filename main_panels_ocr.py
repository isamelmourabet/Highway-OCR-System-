import argparse
import os
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor

from utils_ocr import load_images_dict
from lda_normal_knn_classifier import LdaNormalKNNClassifier
from ocr_classifier import OCRClassifier


def preprocesar_panel(imagen_bgr):
    # 1. Convertir a gris para el clasificador y para el threshold
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Invertir para el threshold: letras blancas → negras, fondo oscuro → claro
    imagen_gris_invertida = cv2.bitwise_not(imagen_gris)

    # 2. Umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        imagen_gris_invertida,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 4. REDUCIR RUIDO: Filtro de mediana para "matar" los puntitos blancos
    thresh_limpio = cv2.medianBlur(thresh, 3)

    return thresh_limpio, imagen_gris_invertida

def detectar_caracteres(imagen):
    # 1. FindContours
    contours, _ = cv2.findContours(imagen, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Filtrar por tamaño y aspect ratio
    candidatos_iniciales = []
    h_img, w_img = imagen.shape

    for contorno in contours:
        x, y, w, h = cv2.boundingRect(contorno)
        if h == 0:
            continue

        aspect_ratio = w / h
        area = w * h

        # FILTROS GEOMÉTRICOS
        if (w >= 3 and h >= 8 and h < h_img * 0.9 and
                aspect_ratio < 1.5 and aspect_ratio > 0.15 and
                area >= 24 and
                (y + h) < h_img * 0.85 and  # Ignoramos el 15% inferior
                x > w_img * 0.05 and  # Ignorar el 5% izquierdo (marco)
                (x + w) < w_img * 0.95):  # Ignorar el 5% derecho (marco)

            candidatos_iniciales.append((x, y, w, h))

    # 3. BORRAR CAJAS DOBLES (Agujeros de la O, P, e, d...)
    candidatos_finales = []

    for i in range(len(candidatos_iniciales)):
        x1, y1, w1, h1 = candidatos_iniciales[i]
        es_agujero_interno = False

        for j in range(len(candidatos_iniciales)):
            if i == j:
                continue
            x2, y2, w2, h2 = candidatos_iniciales[j]

            # Si la caja 1 está completamente dentro de las coordenadas de la caja 2, es un agujero
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                es_agujero_interno = True
                break

        # Si no es un agujero interno, nos la quedamos como letra válida
        if not es_agujero_interno:
            candidatos_finales.append((x1, y1, w1, h1))

    #4. Devolver lista
    return candidatos_finales

def agrupar_por_lineas(candidatos):
    lineas = []
    puntos = [(x + w // 2, y + h // 2, x, y, w, h) for x, y, w, h in candidatos]

    # EVITAR DIAGONALES
    # Esta función comprueba la pendiente (inclinación) de la recta candidata
    def es_linea_horizontal(estimator, X, y):
        return np.abs(estimator.coef_[0]) < 0.2

    while len(puntos) > 2:
        X_centros = np.array([[p[0]] for p in puntos])
        Y_centros = np.array([p[1] for p in puntos])
        try:
            ransac = RANSACRegressor(
                residual_threshold=12.0,
                is_model_valid=es_linea_horizontal # Aplicar el filtro de inclinación
            )
            ransac.fit(X_centros, Y_centros)
            inliers = ransac.inlier_mask_
        except:
            break

        linea_actual = [puntos[i] for i in range(len(puntos)) if inliers[i]]
        puntos = [puntos[i] for i in range(len(puntos)) if not inliers[i]]

        linea_actual = sorted(linea_actual, key=lambda p: p[0])
        lineas.append(linea_actual)

    # Rescatar las letras sueltas que RANSAC ha ingnorado
    if len(puntos) > 0:
        linea_restante = sorted(puntos, key=lambda p: p[0])
        lineas.append(linea_restante)

    lineas = sorted(lineas, key=lambda l: np.mean([p[1] for p in l]))

    return lineas

def leer_panel(imagen, lineas, clasificador):
    lineas_texto = []

    for linea in lineas:
        texto_linea = ""
        for centro_x, centro_y, x, y, w, h in linea:
            # recortar el carácter de la imagen
            char_img = imagen[y:y + h, x:x + w]

            # clasificar
            label = clasificador.predict(char_img)
            char = clasificador.label2char(label)
            texto_linea += char

        lineas_texto.append(texto_linea)

    # unir líneas con '+'
    return "+".join(lineas_texto)

def visualizar_ocr(imagen, imagen_gris, lineas, clasificador):
    # Imagen intermedia: gris con cajas y líneas
    img_cajas_lineas = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

    # Imagen final: color original solo con el texto predicho
    if len(imagen.shape) == 2:
        img_final = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        img_final = imagen.copy()

    h_img, w_img = imagen_gris.shape  # Sacar el ancho total de la imagen

    for linea in lineas:
        if len(linea) == 0:
            continue

        # DIBUJAR LA LÍNEA DE LADO A LADO
        if len(linea) > 1:
            # Coger el primer y último punto para sacar la pendiente (m)
            x1, y1 = linea[0][0], linea[0][1]
            x2, y2 = linea[-1][0], linea[-1][1]

            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
            else:
                m = 0

            # Calcular la altura 'y' en los extremos de la pantalla (x=0 y x=w_img)
            y_inicio = int(y1 - m * x1)
            y_fin = int(y1 + m * (w_img - x1))

            # Dibujar cruzando toda la imagen
            cv2.line(img_cajas_lineas, (0, y_inicio), (w_img, y_fin), (255, 255, 0), 1)
        else:
            # Si la línea solo tiene una letra, pintamos una recta horizontal pura
            y1 = linea[0][1]
            cv2.line(img_cajas_lineas, (0, y1), (w_img, y1), (255, 255, 0), 1)

        # DIBUJAR CAJAS Y TEXTO
        for centro_x, centro_y, x, y, w, h in linea:
            # Caja verde solo en la de debug
            cv2.rectangle(img_cajas_lineas, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Predecir carácter
            char_img = imagen_gris[y:y + h, x:x + w]
            label = clasificador.predict(char_img)
            char = clasificador.label2char(label)

            # Texto verde brillante solo en la imagen final limpia
            cv2.putText(img_final, char, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return img_cajas_lineas, img_final

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    # despues de probar quitar esta parte ya que no se pide en la entrega, es mas para hacer debug
    #parser.add_argument(
    # '--visualize_ocr', action='store_true', help='Visualizar los resultados interactivamente')

    args = parser.parse_args()

    # Load training data
    train_dict = load_images_dict(args.train_path)

    # Create the OCR classifier
    clasificador = LdaNormalKNNClassifier(ocr_char_size=(25, 25))
    clasificador.train(train_dict)

    # Load testing data
    ficheros = sorted([f for f in os.listdir(args.test_path) if f.endswith(".png")])

    # Evaluate OCR over road panels
    with open("./IMÁGENES_PANELES/test_ocr_panels/resultado.txt", "w") as f:
        for fichero in ficheros:
            img_path = os.path.join(args.test_path, fichero)
            imagen = cv2.imread(img_path)
            h, w = imagen.shape[:2]

            # preprocesar: BGR → binarizada (texto blanco/negro) + gris original
            panel_procesado, imagen_gris = preprocesar_panel(imagen)

            # detectar caracteres (sobre la binarizada)
            candidatos = detectar_caracteres(panel_procesado)

            # agrupar por líneas
            lineas = agrupar_por_lineas(candidatos)

            # leer el texto — pasar la gris para que el clasificador haga su propio threshold
            texto_ocr = leer_panel(imagen_gris, lineas, clasificador)

            img_cajas, img_final = visualizar_ocr(imagen, imagen_gris, lineas, clasificador)

            linea = f"{fichero};0;0;{w};{h};1;1.0;{texto_ocr}\n"
            f.write(linea)

            # despues de probar quitar esta parte ya que no se pide en la entrega, es mas para hacer debug
            # VISUALIZACIÓN (Se activa si el usuario pone --visualize_ocr en la terminal)
            """if args.visualize_ocr:
                img_cajas, img_final = visualizar_ocr(imagen, imagen_gris, lineas, clasificador)

                print(
                    f"Mostrando: {fichero} | Píxeles blancos: {cv2.countNonZero(panel_procesado)} | Candidatos: {len(candidatos)}")

                # Mostrar imágenes en ventanas flotantes
                cv2.imshow(f"Cajas y Lineas - {fichero}", img_cajas)
                cv2.imshow(f"Resultado Final OCR - {fichero}", img_final)

                cv2.waitKey(0)

                # Cerrar las ventanas actuales para no saturar la pantalla en la siguiente iteración
                cv2.destroyAllWindows()"""

        print("Proceso finalizado. Resultados guardados en ./IMÁGENES_PANELES/test_ocr_panels/resultado.txt")





