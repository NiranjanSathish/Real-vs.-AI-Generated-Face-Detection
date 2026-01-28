# Real vs. AI-Generated Face Detection (CS6180 HW1)

This project explores deep learning models to distinguish between real human faces and AI-generated faces. It includes data exploration, baseline MLP and CNN models, data augmentation, transfer learning with MobileNetV2, and a deployed interactive Gradio application.

## 📂 Project Structure

- **`Niranjan_Sathish_CS6180_HW1.ipynb`**: The main Jupyter Notebook containing all parts of the assignment:
  - Part 0: Dataset Setup
  - Part 1: Data Exploration
  - Part 2: MLP Baseline
  - Part 3: CNN Implementation
  - Part 4: Data Augmentation
  - Part 5: Transfer Learning
  - Part 6: Fine-Tuning
  - Part 7: Model Comparison
  - Part 8: Interactive Demo
- **`gradio_app/`**: Standalone directory for the interactive Gradio web application.
  - `app.py`: The application logic.
  - `best_model.keras`: The trained model used for inference.
  - `requirements.txt`: Dependencies specific to the app.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or Google Colab

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook

Open `Niranjan_Sathish_CS6180_HW1.ipynb` in Jupyter Notebook or VS Code and run the cells sequentially. Part 0 will automatically download the dataset.

### Running the Gradio App Locally

1.  Navigate to the app directory:
    ```bash
    cd gradio_app
    ```

2.  Run the app:
    ```bash
    python app.py
    ```

3.  Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.

## 📊 Models Implemented

- **MLP Baseline**: A simple fully connected network.
- **CNN Baseline**: A custom convolutional neural network.
- **Augmented CNN**: CNN trained with data augmentation (rotation, flip, zoom).
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet.
- **Fine-Tuned MobileNetV2**: Optimized pre-trained model for this specific task.

## 🎮 Interactive Demo

The project includes a "Human vs. AI Detector" game where you can test your skills against the trained model!
