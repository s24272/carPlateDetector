import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv

# Load the license plate cascade classifier
plate_cascade = cv2.CascadeClassifier("archive2/indian_license_plate.xml")

# Function to detect license plates in an image
def detect_plate(img, text=""):
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(
        plate_img, scaleFactor=1.2, minNeighbors=7
    )
    for (x, y, w, h) in plate_rect:
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)
    if text != "":
        plate_img = cv2.putText(
            plate_img,
            text,
            (x - w // 2, y - h // 2),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.5,
            (51, 181, 155),
            1,
            cv2.LINE_AA,
        )

    return plate_img, plate

# Function to display an image
def display(img_, title=""):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

# Load and display the input image
img = cv2.imread("archive2/car.jpg")
display(img, "input image")

# Detect and display the license plate in the image
output_img, plate = detect_plate(img)
display(output_img, "detected license plate in the input image")
display(plate, "extracted license plate from the image")

# Function to find contours of characters in the license plate
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread("../images/contour.jpg")

    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if (
                intWidth > lower_width
                and intWidth < upper_width
                and intHeight > lower_height
                and intHeight < upper_height
        ):
            x_cntr_list.append(intX)

            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap="gray")
            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)

    plt.show()
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res

# Function to segment characters in the license plate
def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(
        img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
    plt.imshow(img_binary_lp, cmap="gray")
    plt.show()
    cv2.imwrite("../images/contour.jpg", img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

# Segment and display characters from the detected license plate
char = segment_characters(plate)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(char[i], cmap="gray")
    plt.axis("off")

# Define the image transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Load the training dataset
train_dataset = datasets.ImageFolder(
    root="archive2/data/data/train", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load the validation dataset
val_dataset = datasets.ImageFolder(root="archive2/data/data/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Define the neural network model
class LicensePlateModel(nn.Module):
    def __init__(self):
        super(LicensePlateModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(22, 22), padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(16, 16), padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(8, 8), padding="same")
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(4, 4), padding="same")
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = LicensePlateModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        val_losses.append(val_loss / len(val_loader))
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="micro")
        recall = recall_score(all_labels, all_preds, average="micro")
        f1 = f1_score(all_labels, all_preds, average="micro")
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Save the model and optimizer state
    torch.save(model.state_dict(), "model/license_plate_model_weights.pt")
    torch.save(optimizer.state_dict(), "model/license_plate_optimizer.pt")

    # Save the metrics to CSV files
    with open('training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss'])
        for epoch, loss in enumerate(train_losses, start=1):
            writer.writerow([epoch, loss])

    with open('validation_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Validation Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
        for epoch, (loss, accuracy, precision, recall, f1) in enumerate(zip(val_losses, val_accuracies, val_precisions, val_recalls, val_f1s), start=1):
            writer.writerow([epoch, loss, accuracy, precision, recall, f1])

    # Plot the metrics
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Validation Precision')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_recalls, label='Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Validation Recall')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer)

    # Load the model and optimizer state
    model.load_state_dict(torch.load("model/license_plate_model_weights.pt"))
    optimizer.load_state_dict(torch.load("model/license_plate_optimizer.pt"))


    model.eval()
    # model.train()
