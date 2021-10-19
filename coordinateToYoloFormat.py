import json
import cv2


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


with open("data/train.json", "r") as read_file:
    data = json.load(read_file)
    for element in data:
        licensePlates = element.get("nums")
        name = element.get("file")
        print("info {} processed".format(name))
        fileName = name.replace('train/', '')[:-4]
        image = cv2.imread("data/{}".format(name))
        for licensePlate in licensePlates:
            coordinatesBoundingBox = licensePlate.get("box")
            heightImg, widthImg = image.shape[:2]

            x_min = min([x[0] for x in coordinatesBoundingBox])
            x_max = max([x[0] for x in coordinatesBoundingBox])
            y_min = min([y[1] for y in coordinatesBoundingBox])
            y_max = max([y[1] for y in coordinatesBoundingBox])

            height = y_max - y_min
            width = x_max - x_min
            offsetY = round(height * 0.05)
            offsetX = round(width * 0.05)

            x_min = x_min - offsetX
            x_max = x_max + offsetX
            y_min = y_min - offsetY
            y_max = y_max + offsetY

            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [0, 255, 0], 2)
            # cv2.imshow(name, image)
            # cv2.waitKey(0)
            xCenter = ((x_max + x_min) / 2) / widthImg
            yCenter = ((y_max + y_min) / 2) / heightImg
            width = (x_max - x_min) / widthImg
            height = (y_max - y_min) / heightImg

            xCenter = toFixed(xCenter, 6)
            yCenter = toFixed(yCenter, 6)
            width = toFixed(width, 6)
            height = toFixed(height, 6)

            with open("data/yoloLabel/{}.txt".format(fileName), 'a') as f:
                f.write("0 {} {} {} {}\n".format(xCenter, yCenter, width, height))

    nameClasses = "license-plate"
    with open("data/yoloLabel/classes.txt", 'a') as classes:
        classes.write("{}".format(nameClasses))
