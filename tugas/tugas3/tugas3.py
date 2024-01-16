import cv2
import numpy as np

img = cv2.imread("examples/images/detect_blob.png")

imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


canny = cv2.Canny(imgGray, 10, 20)  

contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    if area > 10:
        color = (255, 0, 255)  
        thickness = 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        cv2.drawContours(img, [contour], -1, (0, 255, 255), 2)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.putText(img, "Points: " + str(len(approx)), (x + w + 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.3 , (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Area: " + str(int(area)), (x + w + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX , 0.3 , (255, 0, 255), 1, cv2.LINE_AA)


cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()