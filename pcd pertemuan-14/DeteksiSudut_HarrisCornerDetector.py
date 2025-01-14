import cv2
import numpy as np

# Baca gambar
image = cv2.imread('Foto pake almet PMII.jpg')

# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Konversi ke float32
gray = np.float32(gray)

# Deteksi sudut dengan Harris Corner Detector
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Perbesar hasil agar lebih terlihat
corners = cv2.dilate(corners, None)

# Buat mask untuk piksel yang melebihi threshold
threshold = 0.01 * corners.max()
mask = corners > threshold

# Terapkan warna merah hanya pada piksel yang sesuai
image[mask] = [0, 0, 255]

# Tampilkan hasil
cv2.imshow('Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
