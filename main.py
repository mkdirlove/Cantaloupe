import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix

# Create the parser
parser = argparse.ArgumentParser(description='Predict the ripeness of a fruit')

# Add an argument
parser.add_argument('image', type=str, help='Path to the fruit image')

# Add an argument for the ground truth label
parser.add_argument('ground_truth', type=str, help='Ground truth label (ripe or unripe)')

# Parse the arguments
args = parser.parse_args()

# Load the saved Keras-based KNN model
knn_model = load_model('fruit_knn_model.h5')

# Read the new image
img = cv2.imread(args.image)

# Resize the image to match the expected input shape of the model
img = cv2.resize(img, (100, 100))

# Color segmentation
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_color = np.array([0, 50, 50])  # Define lower color threshold (adjust as needed)
upper_color = np.array([30, 255, 255])  # Define upper color threshold (adjust as needed)
mask = cv2.inRange(hsv_img, lower_color, upper_color)
segmented_img = cv2.bitwise_and(img, img, mask=mask)

# Preprocess the segmented image
segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)
segmented_img = segmented_img.flatten()
segmented_img = np.expand_dims(segmented_img, axis=0)  # Add an extra dimension to match the model input shape

# Predict the ripeness
ripe_probability = knn_model.predict(segmented_img)[0]

# Threshold for considering an image to be similar to the ones in the dataset
threshold = 0.6

if ripe_probability < threshold:
    prediction = "unripe"
else:
    prediction = "ripe"

ground_truth = args.ground_truth

# Calculate accuracy
accuracy = accuracy_score([ground_truth], [prediction])

# Create confusion matrix
labels = ["unripe", "ripe"]
confusion_mat = confusion_matrix([ground_truth], [prediction], labels=labels)

print('The fruit is predicted as', prediction)
print('Ground Truth:', ground_truth)
print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_mat)
