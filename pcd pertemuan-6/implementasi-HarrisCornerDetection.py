import cv2
import numpy as np

# Membaca gambar dalam grayscale
image = cv2.imread('Foto pake almet PMII.jpg', 0)

# Konversi gambar ke tipe float32
gray = np.float32(image)

# Menerapkan Harris Corner Detector
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilasi untuk menonjolkan sudut yang terdeteksi
dst = cv2.dilate(dst, None)

# Thresholding untuk menandai sudut
image[dst > 0.01 * dst.max()] = [255]

# Menampilkan hasil
cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyALLWindows()