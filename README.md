# Image Classification Project

## 📜 Project Overview
This project demonstrates an end-to-end pipeline for classifying images. It involves preprocessing raw image data, visualizing image samples, building a deep learning model using Convolutional Neural Networks (CNNs) to classify images effectively, evaluating and saving best models.

---

## 📊 Steps in the Workflow

1. **Data Preprocessing**:
   - Removed corrupted or "dodgy" images and ensured file extensions were valid.
   - Rescaled images to standardize pixel intensity values (e.g., normalization).
   - Used `tf.keras.utils.image_dataset_from_directory` to load and prepare images for training, validation, and testing.
   - Visualized image samples using **OpenCV** for quick inspection of data quality.

2. **Train-Test-Validation Split**:
   - Split the dataset into **Training**, **Validation**, and **Test** sets for unbiased model evaluation.
   - Configured appropriate batch sizes for efficient training.

3. **Model Development**:
   - Built a **Sequential Deep Learning Model** featuring:
     - **Convolutional Layers** for feature extraction.
     - **MaxPooling Layers** for spatial dimensionality reduction.
     - **Dense Layers** for classification.
   - Compiled the model with:
     - **Loss Function**: Binary Cross-Entropy (for Binary classification), Categorical Cross-Entropy (for multi-class classification).
     - **Optimizer**: Adam for adaptive learning rates.
     - **Evaluation Metric**: Accuracy.

4. **Evaluation**:
   - Trained the model and evaluated performance using the **Accuracy** metric on validation and test sets.

5. **Saving**:
   - Saved the trained models with best evaluation performance.

---

## 🔍 Results
- Achieved competitive classification accuracy on the test dataset.
- Visualized training history (loss and accuracy curves) to monitor model performance and avoid overfitting.

---

## 🛠️ Tools and Technologies
- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib
- **Modeling Framework**: tf.keras (Sequential API)

---

## 📌 Key Learnings
- Importance of cleaning and preprocessing image datasets to ensure data quality.
- Leveraged **CNN architectures** for feature extraction, leading to improved model performance.
- Understood the role of visualizing input data and learning curves for better debugging and optimization.

---

## 🚧 Future Enhancements
1. Explore **data augmentation** techniques to improve model robustness.
2. Experiment with **transfer learning** using pre-trained models like ResNet or VGG.
3. Implement early stopping or learning rate scheduling for better generalization.
4. Deploy the trained model as a web application for real-time predictions.

---

## 🖼️ Visualizations
- **Sample Images**: Visualized a few training and validation samples using OpenCV.
- **Training Metrics**: Plotted accuracy and loss curves to monitor training progress.
