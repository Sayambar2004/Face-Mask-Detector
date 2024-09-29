import streamlit as st
import pickle
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection App",
    page_icon="😷",  # Add your favicon file here
    layout="wide",  # Full-width layout for a cooler look
    initial_sidebar_state="expanded"
)

# Load the saved model using pickle
def load_model():
    with open('mask_detector_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess the uploaded image
def preprocess_image(img, target_size=(128, 128)):
    # Convert the image to RGB format in case it's not
    img = img.convert('RGB')
    
    # Resize the image to the required input size of the model
    img = img.resize(target_size)
    
    # Convert the image to an array
    img_array = np.array(img)
    
    # Normalize the image (scale pixel values to [0, 1])
    img_array = img_array / 255.0
    
    # Reshape the array to match the input shape (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Main function for Streamlit app
def main():
    st.title("Face Mask Detection App")
    st.write("Upload an image to check if the person is wearing a mask or not.")

    # Load the pre-trained model
    model = load_model()

    # Sidebar for developer info and disclaimer
    st.sidebar.title("Developer Info")
    st.sidebar.write("**Developer:** Sayambar Roy Chowdhury")
    st.sidebar.write("**LinkedIn:** [Sayambar Roy Chowdhury](https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/)")
    st.sidebar.write("**GitHub:** [Sayambar](https://github.com/Sayambar2004)")
    st.sidebar.markdown("---")
    st.sidebar.title("Disclaimer")
    st.sidebar.write(
        "This model is built as a personal project and should not be used for other purposes. "
        "The model has an accuracy of **97%**, so it is not the most accurate and may give incorrect predictions."
    )

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and display the image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # Button for making prediction
        if st.button('Predict'):
            # Preprocess the image
            preprocessed_image = preprocess_image(img)

            # Make a prediction
            prediction = model.predict(preprocessed_image)
            predicted_class = (prediction > 0.5).astype("int32")[0][0]

            # Map the predicted class to a label
            labels = {0: "Without Mask", 1: "With Mask"}
            st.write(f"Prediction: **{labels[predicted_class]}**")

    # Description of the model architecture (Always visible)
    st.markdown("---")
    st.subheader("Model Information")
    st.write("""
    This mask detection model is built using Convolutional Neural Networks (CNN). The architecture consists of:
    
    - **Input Layer**: 128x128x3 RGB images.
    - **Three Convolutional Layers**: 
      - First Conv2D layer with 32 filters and a kernel size of 3x3, followed by MaxPooling and 25% dropout.
      - Second Conv2D layer with 64 filters and a kernel size of 3x3, followed by MaxPooling and 25% dropout.
      - Third Conv2D layer with 128 filters and a kernel size of 3x3, followed by MaxPooling and 25% dropout.
    - **Flatten Layer**: Converts the 3D output of the last Conv2D layer into a 1D array.
    - **Dense Layer**: A fully connected Dense layer with 128 units and ReLU activation, followed by 50% dropout.
    - **Output Layer**: A Dense layer with 1 unit and Sigmoid activation for binary classification.
    
    ### Additional Technical Details:
    - **Activation Function**: ReLU (Rectified Linear Unit) is used in hidden layers to introduce non-linearity. Sigmoid activation is used in the final layer for binary classification.
    - **Dropout**: Dropout layers are added after each pooling and dense layer to prevent overfitting. Dropout rates are 25% for convolutional layers and 50% before the output layer.
    - **Loss Function**: Binary Crossentropy is used because this is a binary classification problem (mask/no mask).
    - **Optimizer**: Adam optimizer is used for efficient gradient descent with a learning rate of 0.001.
    - **Early Stopping**: Early stopping was applied to monitor validation loss and prevent overfitting. Training stops if the validation loss doesn’t improve after 5 consecutive epochs.
    - **Data Augmentation**: To improve model robustness, the training data was augmented with random rotations, width/height shifts, shear, zoom, and horizontal flips.
    
    ### Model Performance:
    - **Accuracy**: 95.15% on the training data, and 97.02% on the validation set.
    - **Loss**: The final training loss was 0.1434, while the validation loss was 0.0948.
    
    ### Training Process:
    - The model was trained for 20 epochs using the Adam optimizer and early stopping to prevent overfitting. Data augmentation helped to generalize the model by introducing variations in the training data.
    """)

if __name__ == '__main__':
    main()
