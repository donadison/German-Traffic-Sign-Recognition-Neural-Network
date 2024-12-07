import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Wczytaj wyuczony model
model = load_model('newmodel.h5')

# Zainicjuj kamerę (kamera 0 oznacza domyślną kamerę)
cap = cv2.VideoCapture(0)

# Parametry dla klasyfikacji
image_size = (30, 30)  # rozmiar obrazu, który będzie używany do klasyfikacji

# Mapowanie numerów klas na nazwy znaków drogowych
# Przykładowe nazwy klas; w rzeczywistości musisz dostosować tę listę
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

# Funkcja do wykrywania znaków drogowych i rysowania prostokątów
def detect_traffic_sign(frame):
    # Konwertuj obraz do skali szarości (jeśli chcesz używać detekcji krawędzi lub podobnych technik)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Zastosowanie detekcji krawędzi (opcjonalne, może pomóc w wykrywaniu obiektów)
    edges = cv2.Canny(gray, 50, 150)

    # Wykrywanie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iteracja przez kontury w celu znalezienia prostokątów
    for contour in contours:
        # Sprawdzamy, czy kontur jest wystarczająco duży, aby pasować do znaku drogowego
        if cv2.contourArea(contour) > 500:  # Możesz dostosować tę wartość
            # Określenie prostokąta wokół konturu
            x, y, w, h = cv2.boundingRect(contour)
            
            # Wycięcie fragmentu obrazu zawierającego możliwy znak drogowy
            roi = frame[y:y+h, x:x+w]
            
            # Przekształcenie obrazu na odpowiedni rozmiar
            roi_resized = cv2.resize(roi, image_size)
            roi_resized = np.array(roi_resized) / 255.0  # Normalizacja
            roi_resized = roi_resized.reshape(1, 30, 30, 3)  # Dopasowanie do wymiarów wejściowych modelu
            
            # Predykcja klasy znaku drogowego
            pred = model.predict(roi_resized)
            label = np.argmax(pred)
            sign_name = class_names[label]  # Pobranie nazwy klasy (znaku drogowego)

            # Rysowanie prostokąta wokół wykrytego znaku drogowego
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Prostokąt w kolorze zielonym
            cv2.putText(frame, f"{sign_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Główna pętla do analizy obrazu z kamery
while True:
    # Odczyt obrazu z kamery
    ret, frame = cap.read()
    
    if not ret:
        print("Nie udało się uzyskać obrazu z kamery.")
        break
    
    # Wywołanie funkcji detekcji i rysowania prostokątów
    frame_with_boxes = detect_traffic_sign(frame)
    
    # Wyświetlanie obrazu z naniesionymi prostokątami
    cv2.imshow('Traffic Sign Detection', frame_with_boxes)
    
    # Zatrzymanie pętli, jeśli użytkownik naciśnie 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów kamery i zamknięcie okna
cap.release()
cv2.destroyAllWindows()
