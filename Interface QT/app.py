import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from cartoonizer_ui import Ui_MainWindow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import cv2
import numpy as np


class CartoonizerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CartoonizerApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect buttons to respective functions
        self.ui.pushButton.clicked.connect(self.browse_image)
        self.ui.pushButton_2.clicked.connect(self.generate_cartoon)
        self.ui.pushButton_3.clicked.connect(self.save_image)

        # Attributes for storing paths and images
        self.image_path = None
        self.cartoonized_image = None

        # Load the pre-trained GAN model
        self.model = tf.keras.models.load_model("../pix2pix_generator_model_with metrics.h5", compile=False)

    def browse_image(self):
        # Open a file dialog to select an image
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.ui.label_3.setPixmap(pixmap.scaled(self.ui.label_3.width(), self.ui.label_3.height(), aspectRatioMode=1))

    def preprocess_image(self, image_path, size=(256, 256)):
        # Preprocess the image for the GAN model
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def generate_cartoon(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select an image first!")
            return

        try:
            # Update progress bar
            self.ui.progressBar.setValue(50)

            # Preprocess and generate cartoon
            input_image = self.preprocess_image(self.image_path)
            generated_image = self.model(input_image, training=False).numpy()  # Convert to NumPy
            self.cartoonized_image = generated_image[0] * 255  # Scale to [0, 255]
            self.cartoonized_image = np.clip(self.cartoonized_image, 0, 255).astype(np.uint8)

            # Display the cartoonized image
            height, width, channel = self.cartoonized_image.shape
            bytes_per_line = 3 * width
            cartoonized_qimage = QtGui.QImage(
                self.cartoonized_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(cartoonized_qimage)
            self.ui.label_2.setPixmap(pixmap.scaled(self.ui.label_2.width(), self.ui.label_2.height(), aspectRatioMode=1))

            # Complete progress
            self.ui.progressBar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")



    def save_image(self):
        if self.cartoonized_image is None:
            QMessageBox.warning(self, "No Cartoonized Image", "Please generate a cartoonized image first!")
            return

        # Save the cartoonized image
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                # Use OpenCV to save the image
                cv2.imwrite(file_name, cv2.cvtColor(self.cartoonized_image, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "Image Saved", f"Image saved to: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = CartoonizerApp()
    mainWindow.show()
    sys.exit(app.exec_())
