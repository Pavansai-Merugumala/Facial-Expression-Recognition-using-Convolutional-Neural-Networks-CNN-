
# Facial Expression Recognition using Convolutional Neural Networks (CNN)

This project implements a real-time facial expression recognition system using a Convolutional Neural Network (CNN) in Python with Keras and OpenCV.

## Overview

The goal of this project is to recognize facial expressions (such as Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised) in real-time using a webcam feed. The system uses computer vision techniques for face detection and a trained CNN model to predict the emotion displayed on each detected face.

## Key Features

- Face detection using Haar cascades.
- Multi-class classification of facial expressions using a CNN model.
- Real-time prediction of facial expressions from webcam feed.
- Trained model included (`model.h5`) for immediate use.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- Keras (`keras`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

Install the required Python packages using pip:

```bash
pip install opencv-python keras numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/facial-expression-recognition.git
cd facial-expression-recognition
```

2. Run the script with the `display` mode to start the facial expression recognition system:

```bash
python facial_expression_recognition.py --mode display
```

3. Press `Q` to exit the program.

## Training (Optional)

If you want to retrain the model or use a different dataset:

1. Place your training and validation data in the `data/train` and `data/test` directories, respectively.
2. Change the parameters (e.g., `num_epoch`, `batch_size`, `optimizer`) in the script as needed.
3. Run the script with the `train` mode to train and save the updated model:

```bash
python facial_expression_recognition.py --mode train
```

## Acknowledgments

This project is based on the tutorial by [author's name] on [link to tutorial], with modifications and enhancements for real-time application.

## License

This project is licensed under the [MIT License](LICENSE).
