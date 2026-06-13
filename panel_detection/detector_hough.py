import cv2
import numpy as np

class DetectorHough:
    def detectar(self, imagen):
        # 1. Crear máscara HSV azul
        imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        imagenT = cv2.inRange(imagenHSV, np.array([90, 150, 70]), np.array([150, 255, 255]))
        # imagenT = cv2.inRange(imagenHSV, np.array([90, 100, 70]), np.array([150, 255, 255]))

        # 2. Dilatación morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imagenK = cv2.dilate(imagenT, kernel, iterations=1)

        # 3. Encontrar contornos
        contornos, _ = cv2.findContours(imagenK, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Filtrar contornos y calcular score
        resultados = []
        alto_img, ancho_img = imagen.shape[:2]
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            ratio = w / h
            # if w > 100 and h > 50 and x >= 0 and y >= 0 and x + w <= ancho_img and y + h <= alto_img:
            if w > 100 and h > 50 and ratio >= 1.05 and ratio <= 5.5 and x >= 0 and y >= 0 and x + w <= ancho_img and y + h <= alto_img:
                region = imagenT[y:y + h, x:x + w]
                score = cv2.countNonZero(region) / (w * h)
                resultados.append((x, y, x + w, y + h, score))

        # 5. Dibujar resultados
        imagenResultado = imagen.copy()
        for x1, y1, x2, y2, score in resultados:
            cv2.rectangle(imagenResultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(imagenResultado, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return imagenResultado, resultados
