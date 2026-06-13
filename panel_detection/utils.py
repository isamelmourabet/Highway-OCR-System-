import cv2
import numpy as np

def calcular_iou(rectA, rectB):
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    ancho = ix2 - ix1
    alto = iy2 - iy1

    if (ancho >= 0 and alto >= 0):
        area = ancho * alto
        areaA = (ax2 - ax1) * (ay2-ay1)
        areaB = (bx2 - bx1) * (by2-by1)
        areaUnion = areaA + areaB - area
        return area / areaUnion

    return 0

def contenido_en(rectA, rectB):
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB

    if bx1 <= ax1 and bx2 >= ax2 and by1 <= ay1 and by2 >= ay2:
        return True

    return False

def aplicar_nms(candidatos):
    resultados = []
    # el algoritmo while que tienes en detector.py
    while len(candidatos) > 0:
        #coger el primero, el de mayor score
        mejor = candidatos[0]
        resultados.append(mejor)

        #eliminar el primero de la lista
        candidatos = candidatos[1:]

        #eliminar los que solapen mucho con 'mejor'
        candidatos = [c for c in candidatos if calcular_iou(mejor[:4], c[:4]) < 0.3 and not contenido_en(mejor[:4], c[:4])]

    return resultados