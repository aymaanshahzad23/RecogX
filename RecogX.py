import streamlit as st
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Load the saved model weights
def load_model_weights():
    input_img = Input(shape=(256, 256, 3))
        
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
        
    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
        
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
        
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
        
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(16)(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
        
    model = Model(input_img, output)
    model.load_weights("meso4_model_weights.weights.h5")
    
    return model

# Load the saved model
meso = load_model_weights()

# Function to make prediction
def predict(image):
    # Convert image to RGB mode (if not already in RGB)
    image = image.convert("RGB")
    
    # Preprocess the image
    image = np.array(image.resize((256, 256)))  # Resize image to (256, 256)
    image = image / 255.0  # Rescale pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict using the loaded model
    prediction = meso.predict(image)[0][0]
    
    return prediction

# Streamlit app
def main():
    # Set page title and configure page layout
    st.set_page_config(
        page_title="RecogX - Deepfake Image Detector",
        page_icon=":detective:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS to style the app
    st.markdown(
        """
        <style>
        .css-17eq0hr {
            font-family: 'Arial', sans-serif;
            color: #333;
            background-color: white; /* Set background color to white */
        }
        .st-bj {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .sidebar .sidebar-content .block-container {
            margin-top: 2rem;
            text-align: center;
        }
        .sidebar .sidebar-content .block-container p {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        .sidebar .sidebar-content .block-container a {
            color: #333;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with contact information
    st.sidebar.title("Aymaan Shahzad")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Contact me:")
    st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aymaanshahzad23/)")
    st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aymaanshahzad23)")
    st.sidebar.markdown("[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:aymaanshahzad23@gmail.com)")
    

    # App title and description
    st.title("RecogX - Deepfake Image Detector")
    st.markdown("Detect deepfake images using MesoNet4")
    st.markdown("---")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        prediction = predict(image)

        # Show prediction result
        st.markdown("---")
        if prediction < 0.5:
            st.error("Prediction: This image is predicted to be a deepfake.")
        else:
            st.success("Prediction: This image is predicted to be real.")
        st.markdown("---")
        st.write("Confidence score:", f"{prediction:.2f}")

if __name__ == "__main__":
    main()
