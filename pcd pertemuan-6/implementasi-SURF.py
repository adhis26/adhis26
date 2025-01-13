import cv2

# Membaca gambar
image = cv2.imread('Foto pake almet PMII.jpg')
resized_image = cv2.resize(image, (500, 800))

# Inisialisasi objek ORB
orb = cv2.ORB_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors, = orb.detectAndCompute(resized_image, None)

# Menggambar keypoints di citra
orb_image = cv2.drawKeypoints(resized_image, keypoints, None)

# Menampilkan hasil
cv2.imshow('ORB ', orb_image)
cv2.waitKey(0)
cv2.destroyALLWindows()