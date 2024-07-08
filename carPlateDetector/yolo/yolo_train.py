import cv2
import shutil
import xml.etree.ElementTree as xet
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Function to extract the number from the filename
def the_number_in_the_string(filename):

    # Find the first occurrence of one or more digits in the filename
    match = re.search(r'(\d+)', filename)

    # If a match is found, return the found number as an integer
    if match:
        return int(match.group(0))
    # If no match is found, return 0
    else:
        return 0

# Path to the dataset directory
dataset_path = 'archive'

# Initialize a dictionary to store labels and image information
labels_dict = dict(
    img_path=[],
    xmin=[],
    xmax=[],
    ymin=[],
    ymax=[],
    img_w=[],
    img_h=[]
)

# Get a list of XML files from the annotations directory
xml_files = glob(os.path.join(dataset_path, 'annotations', '*.xml'))

# Process each XML file, sorted by the number in the filename
for filename in sorted(xml_files, key=the_number_in_the_string):
    info = xet.parse(filename)
    root = info.getroot()

    # Find the 'object' element in the XML file and extract bounding box information
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    # Get the image filename and create the full path to the image
    img_name = root.find('filename').text
    img_path = os.path.join(dataset_path, 'images', img_name)

    # Add the extracted information to the corresponding lists in the dictionary
    labels_dict['img_path'].append(img_path)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

    # Load the image to get its dimensions
    height, width, _ = cv2.imread(img_path).shape
    labels_dict['img_w'].append(width)
    labels_dict['img_h'].append(height)


alldata = pd.DataFrame(labels_dict)

print(alldata.head(3))

# Split the data into training and testing sets
train, test = train_test_split(alldata, test_size=0.1, random_state=42)

# Further split the training set into training and validation sets
train, val = train_test_split(train, train_size=8 / 9, random_state=42)

# Display the number of samples in each set
print(f'''
      len(train) = {len(train)}
      len(val) = {len(val)}
      len(test) = {len(test)}
''')

# Function to create folder structure in YOLO format for the dataset split
def make_split_folder_in_yolo_format(split_name, split_df):

    labels_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'images')

    # Create directories for labels and images
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in split_df.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))

        # Calculate the bounding box coordinates in YOLO format
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        # Save the label in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Copy the image to the images directory
        shutil.copy(row['img_path'], os.path.join(images_path, img_name + img_extension))

    print(f"Created '{images_path}' and '{labels_path}'")

# Create folder structures in YOLO format for training, validation, and test sets
make_split_folder_in_yolo_format("train", train)
make_split_folder_in_yolo_format("val", val)
make_split_folder_in_yolo_format("test", test)

datasets_yaml = '''
path: cars_license_plate_new

train: train/images
val: val/images
test: test/images

# number of classes
nc: 1

# class names
names: ['license_plate']
'''

# Save the contents to the datasets.yaml file
with open('datasets.yaml', 'w') as file:
    file.write(datasets_yaml)

model = YOLO('model/yolov9s.pt')

# Train the model
model.train(
    data='datasets.yaml',  # Path to the dataset configuration file
    epochs=300,  # Number of training epochs
    batch=16,  # Batch size
    imgsz=320,  # Image size (width and height) for training
    cache=True  # Cache images for faster training
)

# Find the latest training log directory
log_dir = max(glob('../../results/yolo_results/runs/detect/train*'), key=os.path.getmtime)

results = pd.read_csv(os.path.join(log_dir, 'results.csv'))

results.columns = results.columns.str.strip()

# Extract epochs and accuracy metrics
epochs = results.index + 1  # Add 1 to make indexes match epoch numbers

# Plot results
plt.figure(figsize=(10, 5))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'precision.png'))
plt.show()

# Save the trained model
model.save('yolo_pretrained_model_by300epochs.pt')
