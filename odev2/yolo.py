import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Change to the specific model you're using (e.g., yolov8s.pt)

# Path to the image
image_path = 'C:/Users/Demirr/Desktop/bilgisayragör/images.jpeg'

# Perform inference on the image
results = model.predict(image_path)

# Get the image with bounding boxes and labels
annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the image

# Display the image using OpenCV
cv2.imshow('YOLOv8 Detection', annotated_frame)

# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
