# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset Iris
iris = datasets.load_iris()
X = iris.data  # Fitur
y = iris.target  # Target

# 2. Membagi data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Membuat model KNN (K-Nearest Neighbors)
# Menggunakan k=3 sebagai parameter untuk KNN
model = KNeighborsClassifier(n_neighbors=3)

# 4. Melatih model menggunakan data pelatihan
model.fit(X_train, y_train)

# 5. Memprediksi hasil pada data uji
y_pred = model.predict(X_test)

# 6. Mengevaluasi akurasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. (Opsional) Visualisasi hasil
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o', edgecolors='k', s=100)
plt.title('Scatter plot of Iris dataset (Test set) with KNN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
