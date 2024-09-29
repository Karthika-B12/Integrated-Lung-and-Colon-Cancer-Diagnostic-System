# Integrated-Lung-and-Colon-Cancer-Diagnostic-System


## Project Overview

This project focuses on the detection and classification of lung and colon cancer using histopathology images from the LC25000 dataset. By leveraging state-of-the-art deep learning techniques, the system classifies histopathology images into multiple cancer categories. The project uses two key architectures for this task: **Convolutional Neural Networks (CNNs)** and **EfficientNetB3**. Various variations of these models were experimented with to improve accuracy and generalization, and confusion matrices were generated to evaluate the classification performance.

## Key Achievements
- **Accuracies:** 
  - **CNN Variation 1:** 97.8%
  - **CNN Variation 2:** 71.7%
  - **EfficientNetB3 Variation 1:** 99.4%
  - **EfficientNetB3 Variation 2:** 99.5%
  - **EfficientNetB3 Variation 3:** 99.3%
  
- Applied **data augmentation** and **hyperparameter fine-tuning** using TensorFlow and Keras to enhance model performance.

---

## Models Overview

### 1. Convolutional Neural Networks (CNN)
We implemented two variations of CNN models, fine-tuning hyperparameters such as filter sizes, learning rates, and regularization techniques to optimize performance.

#### **Variation 1:**
- **Hyperparameters:**
  - Filter sizes and kernel sizes were selected for efficient feature extraction.
  - Learning rate: 0.001, batch size: 32.
  - Dropout rate: 0.5.
- **Architecture:**
  - 5 blocks of Conv2D and MaxPooling2D layers for hierarchical feature extraction and downsampling.
  - 3 Dense layers for classification with a softmax activation function for multi-class classification.
- **Activation Function:** ReLU for nonlinearities and mitigating vanishing gradients.
- **Optimizer:** Adamax with a learning rate of 0.001.
- **Accuracy:** 97.8%

#### **Variation 2:**
- **Hyperparameters:**
  - Learning rate: 0.001, batch size: 32.
  - Dropout rate: 0.5, L2 regularization: 0.001.
- **Architecture:**
  - A VGG-like model with 5 blocks of Conv2D, MaxPooling2D, and BatchNormalization layers.
  - 2 Dense layers with LeakyReLU activation and Dropout for regularization.
- **Activation Function:** LeakyReLU (alpha 0.1) to further mitigate vanishing gradients.
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Accuracy:** 71.7%

---

### 2. EfficientNetB3
We used **EfficientNetB3**, a more advanced architecture that provides a balance between performance and efficiency. Three different variations were evaluated, with different dropout rates and regularization settings.

#### **Variation 1:**
- **Hyperparameters:**
  - Dropout rate: 0.45, fixed seed: 123 for reproducibility.
  - L2 regularization (0.016) and L1 regularization (0.006) on Dense layer kernel weights and biases.
- **Architecture:**
  - EfficientNetB3 base model followed by BatchNormalization, Dense layer (256 neurons), Dropout, and softmax output layer.
- **Activation Function:** ReLU for hidden Dense layers.
- **Optimizer:** Adamax optimizer with a learning rate of 0.001.
- **Accuracy:** 99.4%

#### **Variation 2:**
- **Hyperparameters:**
  - Dropout rates: 0.4 and 0.3 for balancing overfitting and model expressiveness.
  - L2 regularization (0.01) on hidden Dense layers.
- **Architecture:**
  - EfficientNetB3 base model followed by Dense layers (512, 256 neurons) and softmax output layer.
- **Activation Function:** ReLU.
- **Optimizer:** Adamax with a learning rate of 0.001.
- **Accuracy:** 99.5%

#### **Variation 3:**
- **Hyperparameters:**
  - Dropout rate: 0.4 for regularization.
  - No explicit weight regularization.
- **Architecture:**
  - EfficientNetB3 base model with 1024 and 512 neurons in Dense layers followed by Dropout and sigmoid output layer.
- **Activation Function:** ReLU in Dense layers, sigmoid in the output layer.
- **Optimizer:** Adam optimizer with a learning rate of 0.0001.
- **Accuracy:** 99.3%

---

## Confusion Matrices

## Model Evaluation

### **Confusion Matrix - CNN (Variation 1):**
![Confusion Matrix CNN Variation 1](https://github.com/user-attachments/assets/5676927b-0f50-48d9-8c69-f5b5f9e866a4)  


### **Confusion Matrix - CNN (Variation 2):**
![Confusion Matrix CNN Variation 2](https://github.com/user-attachments/assets/482bf4a0-b9c3-4e01-917b-7e9d5c603c41)


### **Confusion Matrix - EfficientNetB3 (Variation 1):**
![Confusion Matrix EfficientNetB3 Variation 1](https://github.com/user-attachments/assets/b83273b0-96b3-4e08-8636-96d7f9eec9e3)


### **Confusion Matrix - EfficientNetB3 (Variation 2):**
![Confusion Matrix EfficientNetB3 Variation 2](https://github.com/user-attachments/assets/a4342f70-bcb0-4450-8b30-2ee922c221d7)


### **Confusion Matrix - EfficientNetB3 (Variation 3):**
![Confusion Matrix EfficientNetB3 Variation 3](https://github.com/user-attachments/assets/c60efcf7-8f6a-4f2d-ac89-95f2c4059315)



---

## Conclusion

This project demonstrates the power of deep learning models like CNN and EfficientNetB3 in classifying lung and colon cancer histopathology images. Through rigorous experimentation with multiple model architectures and fine-tuning hyperparameters, we achieved high accuracy, with EfficientNetB3 models reaching up to **99.5% accuracy**.

By incorporating data augmentation, dropout regularization, and careful model design, the system is able to generalize well on unseen data, making it a reliable tool for cancer classification tasks.
