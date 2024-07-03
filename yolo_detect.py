from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt

# Wczytaj wytrenowany model
model = YOLO('runs/detect/train24/weights/last.pt')

def predict_and_plot(path_test_car):
    # Perform prediction on the test image using the model
    results = model.predict(path_test_car, device='cpu')
    # Load the image using OpenCV
    image = cv2.imread(path_test_car)
    # Convert the image from BGR (OpenCV default) to RGB (matplotlib default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['en'])

    # Extract the bounding boxes and labels from the results
    for result in results:
        for box in result.boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get the confidence score of the prediction
            confidence = box.conf[0]

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the confidence score near the bounding box
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Crop the bounding box from the image for OCR
            roi = gray_image[y1:y2, x1:x2]

            # Perform OCR on the cropped image
            text = reader.readtext(roi)
            if len(text) > 0:
                text = text[0][1]
            cv2.putText(image, f'{text}', (x1, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 255), 2)
            print(f"Detected text: {text}")

    # Plot the image with bounding boxes
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()  # Display the image


predict_and_plot('datasets/cars_license_plate_new/test/images/Cars203.png')
