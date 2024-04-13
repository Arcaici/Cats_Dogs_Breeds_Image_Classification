# Breed Classifier

This repository contains code for training and evaluating breed classification models using PyTorch. Two custom CNN models with a hierarchical local approach and a global approach are trained and compared with a pre-trained VGG16 model. The code also includes functionalities for generating heatmaps to visualize model activations.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- seaborn

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/breed-classifier.git
## Usage

1. **Loading Dataset**: The e Oxford-IIIT-Pet dataset can be find inside [kaggle](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset) . The dataset is loaded from the provided annotations file. Annotations are preprocessed to extract necessary information.

2. **Data Augmentation and Transformation**: Training, validation, and test datasets are created with appropriate transformations and augmentation.

3. **Model Definition**: Two custom CNN models are defined inside CatsDogsImageClassification and inside CatsDogsHierarchicalClassification. Model architectures include convolutional layers followed by fully connected layers for breed classification.

4. **Training and Validation**: Models are trained using the provided training and validation datasets. Training loss and accuracy are monitored to ensure model convergence.

5. **Testing**: Trained models are evaluated on the test dataset to measure their performance. Confusion matrices are generated to visualize classification results.

6. **Visualization**: Loss trends during training, as well as model activations (heatmaps), are visualized for analysis.

## Results

- **BreedClassifier Accuracy**: [Test Accuracy 1]
  ![Confusion Matrix - Breed Classifier](https://github.com/Arcaici/Cats_Dogs_Breeds_Image_Classification/blob/main/best_results/128x128_batch_128_dropout_BEST/confusion_matrix_2.png)

- **Hierarchically BreedClassifier Accuracy**: [Test Accuracy 2]
  ![Confusion Matrix - Breed Classifier 2](https://github.com/Arcaici/Cats_Dogs_Breeds_Image_Classification/blob/main/best_results/128x128_Hierairchical/confusion_matrix_breed.png)

- **Pre-trained VGG16 Model Accuracy**: [Test Accuracy VGG16]
  ![Confusion Matrix - VGG16 Classifier](https://github.com/Arcaici/Cats_Dogs_Breeds_Image_Classification/blob/main/best_results/128x128_batch_128_dropout_BEST/confusion_matrix_vgg.png)

## Documentation

For more information please read the documentation present [here](https://github.com/Arcaici/Cats_Dogs_Breeds_Image_Classification/blob/main/Docs/Cats_and_Dogs_Breed_Classification_Using_CNN_and_Transfer_Learning.pdf)
