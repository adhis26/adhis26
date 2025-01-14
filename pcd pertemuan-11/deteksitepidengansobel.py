import cv2
import numpy as np

# Baca Citra grayscale
image = cv2.imread('Foto pake almet PMII.jpg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi menggunakan Sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) # Sobel X
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) # Sobel Y

# Tampilkan hasil
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()