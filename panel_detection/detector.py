import cv2
import numpy as np
from . import utils

class DetectorMSER:
    def detectar(self, imagen):
        imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(delta=5, min_area=800, max_area=60000)
        regiones, rectangulos = mser.detectRegions(imagenGris)
        print(len(regiones))

        imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        # imagenT = cv2.inRange(imagenHSV, np.array([90, 150, 70]), np.array([150, 255, 255]))
        imagenT = cv2.inRange(imagenHSV, np.array([90, 105, 50]), np.array([150, 255, 255]))

        imagenResultado = imagen.copy()

        n = 0

        candidatos = []

        for x, y, w, h in rectangulos:
            region = imagenT[y:y + h, x:x + w]
            nBlancos = cv2.countNonZero(region)
            proporcion = nBlancos / (w * h)

            if w / h >= 1.01 and w / h <= 5.5 and proporcion > 0.2 and (w > 90 and h > 40):
                # if proporcion > 0.5 and (w > 90 and h > 40):
                candidatos.append((x, y, x + w, y + h, proporcion))

        candidatos = sorted(candidatos, key=lambda c: c[4], reverse=True)

        resultados = utils.aplicar_nms(candidatos)

        for x1, y1, x2, y2, score in resultados:
            print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, w:{x2 - x1}, h:{y2 - y1}, score:{score}")

            margen_x = int((x2 - x1) * 0.1)
            margen_y = int((y2 - y1) * 0.1)

            x1 = x1 - margen_x
            y1 = y1 - margen_y
            x2 = x2 + margen_x
            y2 = y2 + margen_y

            cv2.rectangle(imagenResultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(imagenResultado, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            n = n + 1

        return imagenResultado, resultados