import os
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
import re
import pytesseract
from PIL import Image
from ultralytics import YOLO


# Funkcja do ekstrakcji liczby z nazwy pliku
def the_number_in_the_string(filename):
    """
    Wyodrębnia pierwszą sekwencję cyfr z podanego ciągu nazwy pliku i zwraca ją jako liczbę całkowitą.
    Jeśli nie znaleziono cyfr, zwraca 0.

    Parameters:
    filename (str): Ciąg wejściowy, w którym szukamy cyfr.

    Returns:
    int: Pierwsza znaleziona sekwencja cyfr w ciągu wejściowym lub 0, jeśli nie znaleziono cyfr.
    """
    # Szukamy pierwszego wystąpienia jednej lub więcej cyfr w nazwie pliku
    match = re.search(r'(\d+)', filename)

    # Jeśli znaleziono dopasowanie, zwracamy znalezioną liczbę jako całkowitą
    if match:
        return int(match.group(0))
    # Jeśli nie znaleziono dopasowania, zwracamy 0
    else:
        return 0


# Ścieżka do katalogu z danymi
dataset_path = 'archive'

# Inicjalizacja słownika do przechowywania etykiet i informacji o obrazach
labels_dict = dict(
    img_path=[],
    xmin=[],
    xmax=[],
    ymin=[],
    ymax=[],
    img_w=[],
    img_h=[]
)

# Pobieramy listę plików XML z katalogu z adnotacjami
xml_files = glob(os.path.join(dataset_path, 'annotations', '*.xml'))

# Przetwarzamy każdy plik XML, posortowany według liczby w nazwie pliku
for filename in sorted(xml_files, key=the_number_in_the_string):
    # Parsujemy plik XML
    info = xet.parse(filename)
    root = info.getroot()

    # Wyszukujemy element 'object' w pliku XML i ekstrahujemy informacje o ramce ograniczającej
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    # Pobieramy nazwę pliku obrazu i tworzymy pełną ścieżkę do obrazu
    img_name = root.find('filename').text
    img_path = os.path.join(dataset_path, 'images', img_name)

    # Dodajemy wyodrębnione informacje do odpowiednich list w słowniku
    labels_dict['img_path'].append(img_path)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

    # Wczytujemy obraz, aby uzyskać jego wymiary
    height, width, _ = cv2.imread(img_path).shape
    labels_dict['img_w'].append(width)
    labels_dict['img_h'].append(height)

# Konwertujemy słownik na obiekt DataFrame za pomocą biblioteki pandas
alldata = pd.DataFrame(labels_dict)

# Wyświetlamy pierwsze trzy wiersze DataFrame
print(alldata.head(3))

# Dzielimy dane na zbiór treningowy i testowy
train, test = train_test_split(alldata, test_size=0.1, random_state=42)

# Dzielimy zbiór treningowy dodatkowo na zbiór treningowy i walidacyjny
train, val = train_test_split(train, train_size=8 / 9, random_state=42)

# Wyświetlamy liczbę próbek w każdym zbiorze
print(f'''
      len(train) = {len(train)}
      len(val) = {len(val)}
      len(test) = {len(test)}
''')


# Funkcja do tworzenia struktury folderów w formacie YOLO dla zbioru danych
def make_split_folder_in_yolo_format(split_name, split_df):
    """
    Tworzy strukturę katalogów dla danego podziału zbioru danych (train/val/test) w formacie YOLO.

    Parameters:
    split_name (str): Nazwa podziału (np. 'train', 'val', 'test').
    split_df (pd.DataFrame): DataFrame zawierający dane dla danego podziału.

    Funkcja tworzy podkatalogi 'labels' i 'images' w katalogu 'datasets/cars_license_plate/{split_name}',
    a następnie zapisuje etykiety i obrazy w formacie YOLO.
    """
    labels_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'images')

    # Tworzymy katalogi dla etykiet i obrazów
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Iterujemy po każdym wierszu w DataFrame
    for _, row in split_df.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))

        # Obliczamy współrzędne ramki w formacie YOLO
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        # Zapisujemy etykietę w formacie YOLO
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Kopiujemy obraz do katalogu images
        shutil.copy(row['img_path'], os.path.join(images_path, img_name + img_extension))

    print(f"Utworzono '{images_path}' i '{labels_path}'")


# Tworzymy struktury katalogów w formacie YOLO dla zbiorów treningowego, walidacyjnego i testowego
make_split_folder_in_yolo_format("train", train)
make_split_folder_in_yolo_format("val", val)
make_split_folder_in_yolo_format("test", test)

# Definiujemy zawartość pliku datasets.yaml
datasets_yaml = '''
path: cars_license_plate_new

train: train/images
val: val/images
test: test/images

# liczba klas
nc: 1

# nazwy klas
names: ['license_plate']
'''

# Zapisujemy zawartość do pliku datasets.yaml
with open('datasets.yaml', 'w') as file:
    file.write(datasets_yaml)

# Wczytujemy model YOLOv9
model = YOLO('yolov9s.pt')

# Trenujemy model
model.train(
    data='datasets.yaml',  # Ścieżka do pliku konfiguracyjnego zbioru danych
    epochs=300,  # Liczba epok treningowych
    batch=16,  # Rozmiar batcha
    imgsz=320,  # Rozmiar obrazu (szerokość i wysokość) do treningu
    cache=True  # Caching obrazów dla szybszego treningu
)

import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Znajdujemy najnowszy katalog z logami treningowymi
log_dir = max(glob('runs/detect/train*'), key=os.path.getmtime)

# Wczytujemy wyniki treningu z pliku CSV
results = pd.read_csv(os.path.join(log_dir, 'results.csv'))

# Usuwamy ewentualne niepotrzebne spacje w nazwach kolumn
results.columns = results.columns.str.strip()

# Ekstrahujemy epoki i metryki dokładności
epochs = results.index + 1  # Dodajemy 1, aby indeksy były zgodne z numeracją epok

# Wykres wyników
plt.figure(figsize=(10, 5))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'precision.png'))
plt.show()

# Zapisujemy wytrenowany model
model.save('yolo_pretrained_model_by300epochs.pt')

# Funkcja do przewidywania i wyświetlania bounding boxów na obrazie testowym
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
predict_and_plot('datasets/cars_license_plate_new/test/images/Cars415.png')
