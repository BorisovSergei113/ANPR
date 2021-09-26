import os
import imutils
import cv2


class HaarCascadeRussianPlate:
    def __init__(self, HaarCascade):
        self.HaarCascade = HaarCascade

    def debugImageShow(self, title, image, waitKey=True):
        cv2.imshow(title, image)
        if waitKey:
            cv2.waitKey(0)

    def LocateRegionLicensePlate(self, image):
        carplate = image.copy()
        # Метод возвращает массив прямоугольников, возможно описывающих номерной знак, и
        # их нижнюю точку (x, y) ширину и высоту этих прямоугольников
        # scaleFactor - величина изменяющая размер окна классификатора Хаара
        # minNeighbors - мин. количество соседних классификаторов для потверждения детектируемого объекта
        carplateRects = self.HaarCascade.detectMultiScale(carplate, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in carplateRects:
            cv2.rectangle(carplate, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.debugImageShow('plate', carplate)
            return carplate
        print("not found")


if __name__ == "__main__":
    haarcascade = cv2.CascadeClassifier('./haarcascade_russian_plate_number.xml')
    HaarCascadeSearcher = HaarCascadeRussianPlate(haarcascade)

    for i in range(1, 29):
        filename = os.path.join(os.path.dirname(__file__), "objects-sources/img_{}.png".format(i))
        print("objects-sources/img_{}.png".format(i))
        image = cv2.imread(filename)
        image = imutils.resize(image, width=800, height=800)

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        HaarCascadeSearcher.LocateRegionLicensePlate(image)
