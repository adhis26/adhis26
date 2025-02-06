import cv2

# Membaca gambar dalam grayscale
image = cv2.imread('Foto pake almet PMII.jpg', 0)

# Menerapkan deteksi tepi Canny
edges = cv2.Canny(image, 100, 200)

# Menampilkan hasil
cv2.imshow('Canny Edge Detecetion', edges)
cv2.waitKey(0)
cv2.destroyALLWindows()