import argparse
import cv2
import os
import numpy as np
from panel_detection import utils
from panel_detection.detector import DetectorMSER
from panel_detection.detector_hough import DetectorHough
from panel_detection.detector_combinado import DetectorCombinado
from utils_ocr import load_images_dict
from lda_normal_knn_classifier import LdaNormalKNNClassifier
from main_panels_ocr import detectar_caracteres, agrupar_por_lineas, leer_panel, visualizar_ocr, preprocesar_panel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')
    parser.add_argument(
        '--visualize_ocr', action='store_true', default=False)
    args = parser.parse_args()

    # Load training data y crear clasificador OCR
    train_dict = load_images_dict(args.train_path)
    clasificador = LdaNormalKNNClassifier(ocr_char_size=(25, 25))
    clasificador.train(train_dict)

    # Create the detector
    if args.detector == "mser":
        det = DetectorMSER()
    elif args.detector == "hough":
        det = DetectorHough()
    else:
        det = DetectorCombinado()

    # Load testing data
    ficheros = sorted(os.listdir(args.test_path))

    # Evaluate detections
    with open("IMÁGENES_PANELES/resultado.txt", "w") as f:
        os.makedirs("resultado_imgs", exist_ok=True)
        for fichero in ficheros:
            if fichero.endswith(".png"):
                imagen = cv2.imread(f"{args.test_path}/{fichero}")
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

                imagen_resultado, detecciones = det.detectar(imagen)
                detecciones = utils.aplicar_nms(detecciones)
                cv2.imwrite(f"resultado_imgs/{fichero}", imagen_resultado)

                for x1, y1, x2, y2, score in detecciones:
                    margen_x = int((x2 - x1) * 0.1)
                    margen_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - margen_x)
                    y1 = max(0, y1 - margen_y)
                    x2 = min(imagen.shape[1], x2 + margen_x)
                    y2 = min(imagen.shape[0], y2 + margen_y)

                    # OCR sobre el panel recortado
                    panel_recortado = imagen[y1:y2, x1:x2]

                    # preprocesar: BGR → binarizada (texto blanco/negro) + gris original
                    panel_procesado, imagen_gris = preprocesar_panel(panel_recortado)

                    # detectar caracteres (sobre la binarizada)
                    candidatos = detectar_caracteres(panel_procesado)

                    # agrupar por líneas
                    lineas = agrupar_por_lineas(candidatos)

                    # leer el texto
                    texto_ocr = leer_panel(imagen_gris, lineas, clasificador)

                    # visualización
                    if args.visualize_ocr:
                        img_cajas, img_final = visualizar_ocr(panel_recortado, imagen_gris, lineas, clasificador)

                        # Mostrar imágenes en ventanas flotantes
                        cv2.imshow(f"Cajas y Lineas - {fichero}", img_cajas)
                        cv2.imshow(f"Resultado Final OCR - {fichero}", img_final)

                        cv2.waitKey(0)

                        # Cerrar las ventanas actuales para no saturar la pantalla en la siguiente iteración
                        cv2.destroyAllWindows()

                    linea = f"{fichero};{x1};{y1};{x2};{y2};1;{round(score, 2)};{texto_ocr}\n"
                    f.write(linea)