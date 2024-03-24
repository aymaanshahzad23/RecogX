# RecogX - Deepfake Image Detector

RecogX is a web application designed to detect deepfake images using the MesoNet4 model. It provides users with a simple interface to upload images and receive predictions on whether the uploaded image is a deepfake or not.

![RecogX Demo](demo.gif)

## Features

- Detects deepfake images using the MesoNet4 model.
- Provides confidence scores for prediction results.
- Supports image upload in JPEG, JPG, and PNG formats.

## Getting Started

To get started with RecogX, follow these steps:

### Prerequisites

- Python 3.6 or later
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your_username/RecogX.git
   ```

2. Navigate to the project directory:

   ```sh
   cd RecogX
   ```

3. Install the required Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Run the Streamlit app:

   ```sh
   streamlit run app.py
   ```

2. Access the app in your web browser at `http://localhost:8501`.

3. Upload an image by clicking the "Upload an image..." button.

4. Wait for the prediction result to appear.

### Example

![Example](example.png)

## Contributing

Contributions are welcome! If you'd like to contribute to RecogX, please follow these guidelines:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The MesoNet4 model implementation is based on the paper: *[MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888)* by Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen.
- Streamlit: [streamlit/streamlit](https://github.com/streamlit/streamlit)

## Contact

For any inquiries or support, please contact [Aymaan Shahzad](mailto:aymaanshahzad23@gmail.com).

