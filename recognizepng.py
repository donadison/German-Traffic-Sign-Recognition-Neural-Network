import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

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

# Funkcja do rozpoznawania znaku z obrazu
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

    # Wynik klasyfikacji
    print(f"Predykowany znak: {sign_name}")

    return sign_name, pred

# Testowanie funkcji na obrazie PNG
image_path = '00086.png'  # Podaj ścieżkę do pliku PNG z obrazem
sign_name, pred = recognize_traffic_sign(image_path)

