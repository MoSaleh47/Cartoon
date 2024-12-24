# Cartoonizer GAN Application

This project implements a GAN-based cartoonizer that transforms real images into cartoon-style images using a pre-trained Pix2Pix model. The application provides a user-friendly GUI built with PyQt5 to easily select, process, and save cartoonized images.

---
## Features

- **Train Pix2Pix GAN**: Includes the code for training a Pix2Pix GAN model using TensorFlow.
- **Cartoonize Images**: Process real images into cartoonized versions using the pre-trained Pix2Pix model.
- **Graphical User Interface**: A PyQt5-based application for browsing, generating, and saving cartoonized images.
- **Metrics**: Evaluate the quality of the generated images using metrics such as SSIM and PSNR.
## Installation

### Prerequisites
1. **Python 3.10.16 or higher** (Anaconda recommended)
2. Required libraries (see below for version details)
3. (Optional) You can choose to download the full dataset that we used to train our model [here](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset).

### Install Required Libraries

```bash
pip install tensorflow==2.10.0
pip install opencv-python==4.10.0
pip install numpy==1.26.4
pip install matplotlib==3.10.0
pip install pyqt5==5.15.2
pip install tensorflow-hub==0.16.1
pip install pillow==11.0.0
```

### Clone the Repository
```bash
git clone https://github.com/your-repo/cartoonizer-gan.git
cd cartoonizer-gan
```

### Usage

1. **Training the GAN Model**  
   - Open the `main.ipynb` Jupyter notebook in the repository.
   - Follow the instructions to train the Pix2Pix GAN model using the training dataset provided in the `food_dataset/fried_food_dataset` directory.
   - After training, the model will be saved as `pix2pix_generator_model_with metrics.h5` in the `models_saved` directory.

2. **Running the Interface**  
   - Navigate to the `Interface QT` folder in the repository.
   - Run the Python application using the following command:
     ```bash
     python app.py
     ```
   - Or alternatively, you can download the [App.exe](https://drive.google.com/drive/folders/1bZr6SH8wRIBLQpEAfLAOV9l-fl3hxn9W?usp=drive_link).

   - This will launch the PyQt-based graphical interface where users can:
     - Browse and select an image.
     - Generate a cartoonized version using the pre-trained model.
     - Save the generated cartoonized image.


3. **Testing the Model Directly**  
   - Use the `main.ipynb` notebook to test the model on a custom image.
   - Replace the `test_image_path` with the path to your image in the notebook.
   - Run the cells to preprocess the image, generate the cartoonized version, and evaluate metrics (SSIM and PSNR).


### Project Structure

The repository is organized as follows:

```bash
.
├───.ipynb_checkpoints        # Auto-saved checkpoints for Jupyter Notebooks
├───food_dataset              # Dataset directory
│   ├───fried_food_dataset    # Contains training and testing datasets
│   │   ├───test              # Testing dataset
│   │   │   ├───cartoonized_images  # Cartoonized test images
│   │   │   └───real_images         # Real test images
│   │   └───train             # Training dataset
│   │       ├───cartoonized_images  # Cartoonized training images
│   │       └───real_images         # Real training images
├───image_to_test_model       # Directory for images used for testing the model
├───Interface QT              # Directory for PyQt5-based application
│   ├───executable            # Contains the compiled `.exe` file (if created)
│   ├───gen_pic               # Directory for generated cartoonized images
│   ├───__pycache__           # Python cache files
│   ├───app.py                    # Main PyQt5 application script
│   ├───cartoonizer_ui.py         # Python code generated from the UI design file
│   └───Beta.ui                   # Qt Designer UI file for the application
│
├───models_saved              # Directory for saved models
│   └───pix2pix_generator_model_with metrics.h5  # Pre-trained Pix2Pix model
├───main.ipynb                # Jupyter Notebook for training and testing the GAN
└───README.md                 # Project documentation
```

## Usage

### Training the GAN Model
To train the Pix2Pix GAN model from scratch:
1. Place your dataset in the `food_dataset/fried_food_dataset/train/real_images` and `food_dataset/fried_food_dataset/train/cartoonized_images` directories.
2. Open `main.ipynb` in Jupyter Notebook.
3. Run all the cells to train the model.
4. The trained model will be saved in the `models_saved` directory as `pix2pix_generator_model_with_metrics.h5`.

### Testing the Model
To test the pre-trained model:
1. Place your test images in the `image_to_test_model` directory.
2. Open `main.ipynb` in Jupyter Notebook.
3. Run the testing section to generate cartoonized versions of the images.
4. The generated images will be saved in the `Interface QT/gen_pic` directory.

### Using the Graphical User Interface
To use the PyQt5 interface:
1. Ensure all dependencies are installed (see Installation section).
2. Run the following command to start the application:
   ```bash
   python Interface QT/app.py
    ```
3. Use the interface to:
    * Browse: Select an image to cartoonize.
    * Generate: Generate a cartoonized version of the image.
    * Save Cartoonized Image: Save the generated image to your local machine.