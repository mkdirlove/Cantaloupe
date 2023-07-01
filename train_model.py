from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import cv2
import os
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Assuming you have a folder with ripe fruit images and another with unripe ones
ripe_dir = 'ripe/'
unripe_dir = 'unripe/'

def get_images_and_labels(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.resize(img, (100, 100))  # Resize images to a consistent shape
            img = img.flatten()
            images.append(img)
            labels.append(label)
    return images, labels

ripe_images, ripe_labels = get_images_and_labels(ripe_dir, 'ripe')
unripe_images, unripe_labels = get_images_and_labels(unripe_dir, 'unripe')

X = np.array(ripe_images + unripe_images)
y = np.array(ripe_labels + unripe_labels)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

# Create a Keras model wrapper for the KNN classifier
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

knn = KerasClassifier(build_fn=create_model, epochs=10, batch_size=16)

# Create a pipeline with a scaler and the KNN classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', knn)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Output the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Save the trained model in Keras format
pipeline.named_steps['knn'].model.save('fruit_knn_model.h5')
