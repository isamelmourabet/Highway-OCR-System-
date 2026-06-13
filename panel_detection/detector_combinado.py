from .detector import DetectorMSER
from .detector_hough import DetectorHough

class DetectorCombinado:
    def detectar(self, imagen):
        _, detecciones_mser = DetectorMSER().detectar(imagen)
        _, detecciones_hough = DetectorHough().detectar(imagen)
        return imagen, detecciones_mser + detecciones_hough