import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r'D:\Projects\DL\Grocery_app\Model\GroceryStoreDataset\fruit_vegetable_model.h5')

# Define the correct labels/classes for fruit detection
# Update the fruit labels to include all 19 classes
fruit_labels = ['apple', 'banana', 'cucumber', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'orange', 'pineapple', 'pomegranate', 'watermelon', 'class_14', 'class_15', 'class_16', 'class_17', 'class_18', 'class_19']


# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize the frame to fit the model's input size
    resized_frame = cv2.resize(frame, (150, 150))  # Adjust based on model input size
    
    # Preprocess the frame: Convert to the format the model expects
    img_array = np.array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if needed

    # Make prediction
    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")  # Debugging: print the raw prediction values
    print(f"Prediction shape: {prediction.shape}")  # Check the shape of the prediction

    # Ensure that prediction is in the expected shape
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted class index: {predicted_class}")  # Debugging: print predicted class index

    # Map the prediction to the label
    try:
        predicted_label = fruit_labels[predicted_class[0]]
    except IndexError:
        print("Error: Prediction index out of range. Check the number of labels and model output.")
        break

    # Display the result
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Fruit Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
