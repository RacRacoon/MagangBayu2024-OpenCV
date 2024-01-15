import cv2 as cv
import numpy as np

# Tuliskan Kodingan kalian dibawah
image = cv.imread('tugas1.png')


if image is None:
    print("Gambar tidak dapat dibaca. Pastikan path file gambar benar.")
    exit()


hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)


lower_green_tosca = np.array([80, 50, 50])
upper_green_tosca = np.array([100, 255, 255])


green_tosca_mask = cv.inRange(hsv_image, lower_green_tosca, upper_green_tosca)


contours, _ = cv.findContours(green_tosca_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Inisialisasi variabel bounding box yang akan menyatukan semua kotak
merged_bbox = None


for contour in contours:
    x, y, w, h = cv.boundingRect(contour)

    # Tentukan faktor perbesaran (misalnya, 1.2)
    scale_factor = 1.2

    # Perbesar bounding box
    x_new = int(x - (scale_factor - 1) * w / 2)
    y_new = int(y - (scale_factor - 1) * h / 2)
    w_new = int(w * scale_factor)
    h_new = int(h * scale_factor)

    
    if merged_bbox is None:
        merged_bbox = (x_new, y_new, w_new, h_new)
    else:
        merged_bbox = (
            min(merged_bbox[0], x_new),
            min(merged_bbox[1], y_new),
            max(merged_bbox[0] + merged_bbox[2], x_new + w_new) - min(merged_bbox[0], x_new),
            max(merged_bbox[1] + merged_bbox[3], y_new + h_new) - min(merged_bbox[1], y_new)
        )


if merged_bbox is not None:
    x, y, w, h = merged_bbox
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv.imshow('Deteksi Objek Hijau Tosca', image)
cv.waitKey(0)
cv.destroyAllWindows()
