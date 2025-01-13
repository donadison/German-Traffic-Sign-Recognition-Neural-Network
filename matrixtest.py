import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Wczytanie wytrenowanego modelu
model = load_model('newmodel.h5')

# Funkcja do przygotowania obrazu do predykcji
def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((30, 30))  # Dopasowanie rozmiaru
    img_array = np.array(img) / 255.0  # Normalizacja
    img_array = img_array.reshape(1, 30, 30, 3)  # Dopasowanie wymiarów
    return img_array

# Funkcja do generowania macierzy błędów z wykorzystaniem pliku z opisami
def evaluate_model_with_labels(test_folder, labels_file):
    # Wczytanie danych z pliku (np. CSV lub Excel)
    df = pd.read_csv(labels_file)
    
    true_labels = []
    predicted_labels = []
    
    # Iterowanie przez wiersze w pliku z opisami
    for _, row in df.iterrows():
        image_path = os.path.join(test_folder, os.path.basename(row['Path']))
        class_id = row['ClassId']  # Prawdziwa etykieta klasy
        
        if os.path.exists(image_path):
            # Przygotowanie obrazu
            img_array = prepare_image(image_path)
            
            # Predykcja modelu
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred)
            
            # Dodanie etykiety prawdziwej i przewidzianej
            true_labels.append(class_id)
            predicted_labels.append(predicted_class)
    
    # Generowanie macierzy błędów
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(43)))

    # Wizualizacja macierzy błędów z kolorowaniem
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Macierz Błędów")
    plt.xlabel("Przewidziane Etykiety")
    plt.ylabel("Prawdziwe Etykiety")
    tick_marks = np.arange(43)
    plt.xticks(tick_marks, tick_marks, rotation=90)
    plt.yticks(tick_marks, tick_marks)

    # Dodanie siatki na wykres
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Ścieżki do danych
test_folder = r"Test"  # Folder z obrazami testowymi
labels_file = r"Test.csv"  # Plik CSV z opisami

# Uruchomienie ewaluacji
evaluate_model_with_labels(test_folder, labels_file)
