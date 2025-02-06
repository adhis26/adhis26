# Import library yang diperlukan
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Memuat dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Pra-pemrosesan data
# Menormalkan data (skala pixel gambar dari 0-255 ke 0-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Mengubah bentuk data menjadi (28, 28, 1) karena gambar MNIST adalah 28x28 piksel dan hanya satu saluran warna (grayscale)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 3. Membangun model CNN
model = models.Sequential([
    # Layer Konvolusi pertama
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Layer Konvolusi kedua
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Layer Konvolusi ketiga
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten layer untuk mengubah gambar menjadi vektor
    layers.Flatten(),
    
    # Fully connected layer
    layers.Dense(64, activation='relu'),
    
    # Output layer untuk klasifikasi 10 kelas (digit 0-9)
    layers.Dense(10, activation='softmax')
])

# 4. Menyusun model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Melatih model CNN
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 6. Menguji akurasi model pada data uji
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# 7. (Opsional) Menampilkan contoh gambar dari data uji
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {model.predict(x_test[0].reshape(1, 28, 28, 1)).argmax()}")
plt.show()
