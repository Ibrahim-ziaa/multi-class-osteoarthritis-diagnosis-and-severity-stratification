
# Multi-Class Osteoarthritis Diagnosis and Severity Stratification

This project leverages deep learning techniques to classify knee X-ray images into multiple severity levels of osteoarthritis (0-5). The system is designed to assist in medical diagnosis and treatment planning by providing a reliable classification of disease progression.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Model Architecture](#model-architecture)
5. [Training Details](#training-details)
6. [Results](#results)
7. [How to Use](#how-to-use)
8. [Future Work](#future-work)

---

## **Overview**
Osteoarthritis is a chronic joint condition that affects millions of people worldwide. This project focuses on creating an automated system to classify X-ray images into distinct severity levels:
- **Level 0**: Healthy
- **Levels 1-4**: Increasing severity of osteoarthritis.

The goal is to enhance diagnostic accuracy and support clinicians in assessing disease progression effectively.

---

## **Dataset**
- **Source**: Knee X-ray dataset containing over 10,000 labeled images.
- **Labels**: Severity levels (0-4).
- **Preprocessing**:
  - Resize images to 224x224 pixels.
  - Normalize pixel values using ImageNet statistics.
  - Apply data augmentation (e.g., rotation, cropping, color jitter).

---

## **Workflow**
1. **Input X-ray Images**:
   - Source: Dataset with labeled severity levels.
2. **Preprocessing**:
   - Resize, normalize, and augment images.
3. **Deep Learning Model**:
   - Fine-tuned EfficientNet-B0 with a custom classifier.
4. **Training**:
   - CrossEntropyLoss and Adam optimizer.
   - Early stopping to prevent overfitting.
5. **Prediction**:
   - Model outputs severity levels (0-4).
6. **Evaluation**:
   - Metrics: Accuracy, loss.

---

## **Model Architecture**
- **Base Model**: EfficientNet-B0 (pretrained on ImageNet).
- **Classifier**:
  - Fully connected layer with 5 output nodes for multi-class classification.
- **Optimization**:
  - Loss Function: CrossEntropyLoss.
  - Optimizer: Adam with weight decay.
  - Scheduler: Cosine annealing learning rate scheduler.

---

## **Training Details**
- **Batch Size**: 32
- **Epochs**: 25
- **Early Stopping**: Triggered after 5 epochs with no improvement in validation loss.
- **Hardware**: Trained on GPU (if available).
- **Metrics**:
  - Training and validation loss.
  - Accuracy over epochs.

---

## **Results**
- **Validation Accuracy**: Achieved ~70% accuracy on validation data.
- **Test Accuracy**: Model evaluated on unseen test data with competitive performance.
- **Visualization**:
  - Training vs. validation loss and accuracy plots.
  - Sample predictions with input images and predicted severity levels.

---

## **How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ibrahim-ziaa/multi-class-osteoarthritis-diagnosis-and-severity-stratification.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Model**:
   - Use the provided script to classify an image:
     ```bash
     python classify_image.py --image_path <path_to_image>
     ```
   - Example:
     ```bash
     python classify_image.py --image_path ./data/sample_xray.png
     ```
4. **Output**:
   - The predicted severity level (0-4) is displayed along with the input image.

---

## **Future Work**
1. **Improved Model Generalization**:
   - Incorporate additional datasets to improve model robustness.
2. **Explainability**:
   - Integrate techniques like Grad-CAM to visualize model focus areas.
3. **Deployability**:
   - Develop a web-based interface for clinicians to upload X-rays and get predictions.

---

## **Contributing**
Contributions are welcome! Please fork the repository and create a pull request with your changes.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
Special thanks to the creators of the dataset and the open-source community for providing excellent tools and resources.

