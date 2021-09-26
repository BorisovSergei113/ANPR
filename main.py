import os
import cv2
import numpy as np
import imutils

from skimage.segmentation import clear_border


class MorphologicalLicenseNumberSearcher:
    def __init__(self, debug=False):
        self.debug = debug
        self.minAR = 4  # отношение длины/ширины российского номера
        self.maxAR = 5

    def debugImageShow(self, title, image, waitKey=True):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def locateLicensePlateCandidates(self, gray, keepCandidates=10):
        # Применям морфологическою опреацию blackhat
        # Blackhat - фильтр, который вычитает исходное изображение из замкнутого.
        # она позволяет выделить яркие объекты
        # расположенные на темном фоне
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))  # ядро 13х5 - повторяет прямоугольную форму знака
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debugImageShow("Blackhat", blackhat)

        # Поиск белых областей на изображении
        # Фильтр - замыкание сначала к изображению применяется дилатация, а потом эрозия.
        # Замыкание удаляет темные детали на изображении, почти не изменяя яркие детали,
        # т.е. сглаживает и удаляет нежелательные шумы.
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugImageShow("Light Regions", light)

        # Градиент Шарра обнаруживает края изображения и подчеркивает границы символов на номерном знаке
        # в направлении оси х (вертикальные линии)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debugImageShow("Scharr", gradX)

        # размываем изображение после градиента шарра для дополнительного очищения изображения от ненужных деталей
        # затем снова применяем замыкание к очищенному изображению и выявляем границы
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugImageShow("Grad Thresh", thresh)

        # выполянем 2 сужения и 2 расширения для очистки от мелких областей, не относящихся к номеру
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debugImageShow("Grad Erode/Dilate", thresh)

        # накладываем light изображение на thresh
        # выполняем расширения и сужение для заполнения получившихся пятен и очистки от мелкого шума
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debugImageShow("Final", thresh, waitKey=True)

        # находим все keepCandidates выделенных контуров
        # сортируем в порядке убывания по их площади
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keepCandidates]
        return cnts

    def locateLicensePlate(self, gray, candidates, clearBorderContours=False):
        finalLicensePlateContour = None
        regionOfInterest = None

        for c in candidates:
            # определяем прямоугольник, ограничивающий найденный контур
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)  # отношения сторон описывающего прямоугольника

            if self.minAR <= ar <= self.maxAR:
                finalLicensePlateContour = c
                licensePlate = gray[y:y + h, x:x + w]
                regionOfInterest = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # если поле возможного нахождения номера касается границ изображения
                # то это скроее всего шум
                if clearBorderContours:
                    regionOfInterest = clear_border(regionOfInterest)

                self.debugImageShow("License Plate", licensePlate)
                self.debugImageShow("regionOfInterest", regionOfInterest, waitKey=True)
                break

        return regionOfInterest, finalLicensePlateContour


class CannyLicenseNumberSearcher:
    def __init__(self):
        self.debug = True
        pass

    def debugImageShow(self, title, image, waitKey=True):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def locateLicensePlate(self, gray, img):
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        self.debugImageShow('gray', gray)
        edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
        self.debugImageShow('edged', edged)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            print("No contour detected")
            return

        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        self.debugImageShow('contour', img)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)
        self.debugImageShow('new-image', new_image)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        self.debugImageShow('cropped', cropped)


if __name__ == "__main__":
    NumberSearcher = MorphologicalLicenseNumberSearcher(debug=True)

    filename = os.path.join(os.path.dirname(__file__), "objects-sources/img_10.png")

    image = cv2.imread(filename)
    NumberSearcher.debugImageShow('car', image)
    image = imutils.resize(image, width=600, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # candidates = NumberSearcher.locateLicensePlateCandidates(gray)
    # NumberSearcher.locateLicensePlate(gray, candidates, True)

    CannyNumberSearcher = CannyLicenseNumberSearcher()
    CannyNumberSearcher.locateLicensePlate(gray, image)
