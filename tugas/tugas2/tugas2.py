import cv2
import numpy as np

# Import image
img = cv2.imread("tugas2.jpg")

# Blur the image
imgBlur = cv2.GaussianBlur(img, (5, 5), 100)

# Define color bounds
lowerBound = np.array([0, 0, 0], dtype=np.uint8)
upperBound = np.array([255, 100, 30], dtype=np.uint8)

# Create a mask
mask = cv2.inRange(imgBlur, lowerBound, upperBound)

# Canny edge detection
canny = cv2.Canny(mask, 10, 25)  # Adjust these thresholds if needed

# Find contours
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)

    if 5000 < area < 100000:
        cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.putText(img, str(len(approx)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX , 5, (0, 0, 0), 2, cv2.LINE_AA) 

# Display images
cv2.imshow("Hasil", img)

cv2.waitKey(0)
cv2.destroyAllWindows()