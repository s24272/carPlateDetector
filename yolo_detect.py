from ipywidgets import interact, widgets
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from ultralytics import YOLO
from glob import glob

# Wczytaj wytrenowany model
model = YOLO('yolo_pretrained_model_by300epochs.pt')

def predict_and_plot(path_test_car):
    """
    Przewiduje i wyświetla bounding boxy na podanym obrazie testowym za pomocą wytrenowanego modelu YOLO.
    """
    # Wykonujemy predykcję na obrazie testowym za pomocą modelu
    results = model.predict(path_test_car, device='cpu')

    # Wczytujemy obraz za pomocą OpenCV
    image = cv2.imread(path_test_car)
    # Konwertujemy obraz z BGR (domyślny format OpenCV) do RGB (domyślny format matplotlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ekstrahujemy bounding boxy i etykiety z wyników predykcji
    for result in results:
        for box in result.boxes:
            # Pobieramy współrzędne bounding boxa
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Pobieramy wynik pewności predykcji
            confidence = box.conf[0]

            # Rysujemy bounding box na obrazie
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Rysujemy wynik pewności obok bounding boxa
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Wycinamy bounding box z obrazu dla pytesseract
            roi = image[y1:y2, x1:x2]

            # Wykonujemy OCR na wyciętym fragmencie
            text = pytesseract.image_to_string(Image.fromarray(roi))
            cv2.putText(image, f'{text}', (x1, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 255), 2)
            print(f"Detected text: {text}")

    # Wyświetlamy obraz z bounding boxami
    plt.imshow(image)
    plt.axis('off')  # Ukrywamy osie
    plt.show()  # Wyświetlamy obraz

# Przewidujemy i wyświetlamy bounding boxy na wybranym obrazie testowym
predict_and_plot('datasets/cars_license_plate_new/test/images/Cars429.png')