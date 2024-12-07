import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image

# Wczytanie wytrenowanego modelu
model = load_model('newmodel.h5')

# Mapowanie numerów klas na nazwy znaków drogowych
class_names = [
    "Speed limit (20 km/h)", "Speed limit (30 km/h)", "Speed limit (50 km/h)", "Speed limit (60 km/h)",
    "Speed limit (70 km/h)", "Speed limit (80 km/h)", "End of speed limit (80 km/h)", "Speed limit (100 km/h)",
    "Speed limit (120 km/h)", "No overtaking", "No overtaking (trucks)", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrian crossing", "Children", "Bicycle crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout", "End of priority road", "Priority road", "No parking", "No stopping"
]

# Funkcja do rozpoznawania znaku drogowego
def recognize_traffic_sign(image_path):
    # Wczytanie obrazu
    img = Image.open(image_path)
    
    # Zmiana rozmiaru obrazu (do rozmiaru, który pasuje do modelu)
    img = img.resize((30, 30))
    
    # Konwersja obrazu na tablicę NumPy
    img_array = np.array(img)

    # Normalizacja obrazu (wszystkie piksele w zakresie 0-1)
    img_array = img_array / 255.0

    # Dopasowanie wymiarów do tego, czego model oczekuje
    img_array = img_array.reshape(1, 30, 30, 3)

    # Predykcja klasy znaku drogowego
    pred = model.predict(img_array)
    label = np.argmax(pred)  # Klasa z najwyższym prawdopodobieństwem
    sign_name = class_names[label]  # Pobranie nazwy klasy (znaku drogowego)

    return sign_name, img_array[0]  # Zwracamy także obraz w formie tablicy NumPy

# Funkcja do przetwarzania obrazów w folderze i dodawania podpisów za pomocą Matplotlib
def process_images_in_folder(folder_path):
    # Przechodzimy po wszystkich obrazach w folderze
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    
    # Tworzymy odpowiednią liczbę wykresów
    n = len(filenames)
    
    # Zmniejszamy liczbę kolumn, aby obrazy były mniejsze i wygodniej mieściły się w oknie
    cols = 3  # Liczba kolumn w siatce (zmniejszy to szerokość siatki)
    rows = (n + cols - 1) // cols  # Liczba wierszy, zaokrąglamy w górę
    
    # Zmniejszamy wielkość całej siatki wykresów
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))  # Zmniejszamy rozmiar wykresu
    axes = axes.flatten()

    for i, filename in enumerate(filenames):
        # Tworzymy pełną ścieżkę do obrazu
        image_path = os.path.join(folder_path, filename)

        # Rozpoznajemy znak drogowy
        sign_name, img_array = recognize_traffic_sign(image_path)

        # Wyświetlamy obraz
        ax = axes[i]
        ax.imshow(img_array)
        
        # Ustawienie napisu
        ax.set_title(sign_name, fontsize=8, fontweight='bold', color='white', backgroundcolor='black')
        
        # Ukrycie osi
        ax.axis('off')

    # Ukrywamy pozostałe puste wykresy, jeśli są
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Wyświetlenie okna z obrazami
    plt.tight_layout()
    plt.show()

# Ścieżka do folderu z obrazami
folder_path = 'settest'  # Ustaw ścieżkę do folderu 'settest'

# Procesowanie obrazów
process_images_in_folder(folder_path)
