# Codealpha_

# Acoustic Emotion Detection using LSTM

A deep learning project that classifies human emotions from audio files using Mel-Frequency Cepstral Coefficients (MFCCs) and a Long Short-Term Memory (LSTM) neural network.


## Project Overview

Recognizing human emotion from speech is a critical task in affective computing, human-computer interaction, and virtual assistants. This project processes raw audio datasets (such as RAVDESS or TESS), extracts audio features via **Librosa**, and trains a sequential **LSTM model** in TensorFlow/Keras to categorize speech into 7 distinct emotional states.


## Key Features

* **Exploratory Audio Analysis:** Generates visual representations of audio data, including waveplots and spectrograms, alongside interactive audio playback.
* **Feature Extraction:** Computes 40-dimensional MFCC features averaged over time to capture temporal and spectral characteristics of speech.
* **Deep Learning Model:** Implements a multi-layer LSTM network combined with Dropout layers for regularization to prevent overfitting.
* **Performance Tracking:** Visualizes training vs. validation accuracy and loss curves over 50 epochs.


## Tech Stack & Libraries

* **Language:** Python 3.11
* **Data Manipulation & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Audio Processing:** `librosa`
* **Deep Learning:** TensorFlow / Keras (`Sequential`, `LSTM`, `Dense`, `Dropout`)
* **Preprocessing:** `scikit-learn` (`OneHotEncoder`)


## Project Structure & Workflow

1. **Environment Setup & Data Loading:** Walks through the `/kaggle/input` directory to collect file paths and parse emotion labels from filenames.
2. **Exploratory Data Analysis (EDA):** Plots class distribution and inspects wave/spectrogram patterns for emotions like `angry`, `happy`, `sad`, `fear`, `disgust`, `neutral`, and pleasant surprised (`ps`).
3. **Feature Extraction (`extract_mfcc`):** Extracts 40 MFCC coefficients from a 3-second audio window with an offset.
4. **Model Architecture:**
   * **LSTM Layer:** 256 units
   * **Dropout Layers:** 20% rate for regularization
   * **Dense Layers:** 128 and 64 units (ReLU activation)
   * **Output Layer:** 7 units (Softmax activation for multi-class classification)
5. **Training & Evaluation:** Compiled using the Adam optimizer and categorical crossentropy loss, achieving a peak validation accuracy around **72.32%**.


## Results & Visualizations

* **Accuracy Curve:** Monitors training and validation accuracy progression across 50 epochs.
* **Loss Curve:** Tracks categorical crossentropy loss to check for model convergence and potential overfitting.

## Future Improvements

* Implement a `ModelCheckpoint` callback to automatically save the best-performing model weights based on validation accuracy.
* Add an adaptive learning rate scheduler (`ReduceLROnPlateau`) to optimize convergence during later epochs.
* Experiment with Bidirectional LSTMs or Convolutional Recurrent Neural Networks (CRNN) for enhanced performance.
