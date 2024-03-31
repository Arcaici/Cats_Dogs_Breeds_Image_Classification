import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from Dataset import CatsDogsDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from HierarchicalDatabase import BreedDataset

# ----------- LOADING DATASET ----------- #

# Loading list.txt
dir_annotation = "./annotations/list.txt"
annotations = pd.read_csv(dir_annotation, skiprows=6, header=None, names=['#Image CLASS-ID SPECIES BREED ID'])
print(annotations.head())

# preprocessing information
annotations[['CLASS-ID', 'SPECIES', 'BREED', 'ID']] = annotations['#Image CLASS-ID SPECIES BREED ID'].str.split(
    expand=True)
annotations = annotations.drop('#Image CLASS-ID SPECIES BREED ID', axis=1)
annotations = annotations.rename(
    columns={"CLASS-ID": "image", "SPECIES": "CLASS-ID", 'BREED': "SPECIES", "ID": "BREED ID"})
annotations[["CLASS-ID", "SPECIES", "BREED ID"]] = annotations[["CLASS-ID", "SPECIES", "BREED ID"]].astype(int)
print(annotations.head())

# adding the extension to image, so it can be used to access the real image
annotations['image'] = annotations['image'].apply(lambda x: str(x) + '.jpg')
annotations = annotations.reset_index()
annotations = annotations.drop('index', axis=1)

# Extracting the classname/breed of the animal
annotations['classname'] = annotations['image'].apply(lambda x: str(x)[:str(x).rindex('_')])

# Adding information about cat or dog based on the 'Species' column to the 'classname' column
annotations['classname'] = annotations.apply(
    lambda row: f"{('dog' if row['SPECIES'] == 2 else 'cat')}_{row['classname']}", axis=1)
print(annotations.head(5))



proj_df, test_df = train_test_split(annotations, test_size=0.3, random_state=42)

train_df, val_df = train_test_split(proj_df, test_size=0.3, stratify=proj_df['classname'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

base_train_dataset = BreedDataset(train_df, transform=transform)
aug_train_dataset = BreedDataset(train_df,  transform=augmentation_transform)
train_dataset = ConcatDataset([base_train_dataset, aug_train_dataset])

val_dataset = BreedDataset(val_df, transform=transform)
test_dataset = BreedDataset(test_df, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

unique_values, counts = np.unique(proj_df['classname'], return_counts=True)
print("# of unique values : ", len(unique_values))
plt.bar(unique_values, counts)
plt.title('Distribution of classname')
plt.xlabel('classname')
plt.ylabel('Count')
plt.xticks(rotation=90)

# plt.show()

# ----------- DEFINING CUSTOM MODELS ----------- #
class CatsDogsClassifier(nn.Module):
    def __init__(self):
        super(CatsDogsClassifier, self).__init__()

        # CNN architecture
        self.species_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # BREED Branch
        self.species_fc = nn.Sequential(
            # nn.Linear(65536, 1024),
            nn.Linear(16384, 1024),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        cnn_output = self.species_cnn(x)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)
        species_output = self.species_fc(cnn_output)

        return species_output

    def get_conv_layers(self):
        return [idx for idx, layer in enumerate(self.modules()) if isinstance(layer, nn.Conv2d)]

    def generate_heatmaps(self, sample_images):
        # Set the model to evaluation mode
        self.cuda().eval()

        activations = []
        hooks = []
        for layer_idx in self.get_conv_layers():
            conv_layer = list(self.modules())[layer_idx]
            hook = conv_layer.register_forward_hook(self.get_activation_hook(activations))
            hooks.append(hook)

        with torch.no_grad():
            self(sample_images)

        for hook in hooks:
            hook.remove()

        # Calculate mean activations and create heatmaps
        mean_activations = [activation.mean(dim=1, keepdim=True) for activation in activations]
        heatmaps = [torch.nn.functional.interpolate(mean_activation, size=sample_images.shape[2:], mode='bilinear',
                                                    align_corners=False) for mean_activation in mean_activations]

        return heatmaps

    def get_activation_hook(self, activations):
        def hook(module, input, output):
            activations.append(output.cpu())

        return hook



# ----------- TRAINING AND VALIDATION CatsDogsClassifier MODELS ----------- #
# Create model
catdog_model = CatsDogsClassifier()
summary_str = summary(catdog_model, input_size= (batch_size, 3, 128, 128))
criterion_species = nn.CrossEntropyLoss()
optimizer_spieces = optim.Adam(catdog_model.parameters(), lr=0.0001, weight_decay=0.001)
# model.load_state_dict(torch.load('catdog_model.pth', map_location=torch.device('gpu')))


# Max num of epochs
num_epochs = 50

# Loss lists
train_loss_list = []
val_loss_list = []

# Accuracy lists
train_species_accuracy_list = []
val_species_accuracy_list = []

if torch.cuda.is_available():
    catdog_model = catdog_model.cuda()

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
no_improvement_count = 0

# Training the model
begin = time.time()
for epoch in tqdm(range(num_epochs), desc="Epochs"):

    correct_species_train = 0
    total_train = 0
    species_loss_sum = 0
    total_samples = 0

    catdog_model.train()

    for inputs, _, species_labels in train_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            species_labels = species_labels.cuda()
        optimizer_spieces.zero_grad()
        species_output = catdog_model(inputs)

        # Move the output in the same device of gender_labels
        species_output = species_output.to(species_labels.device)
        species_labels = species_labels.long()

        # Calculate loss
        species_loss = criterion_species(species_output, species_labels)
        total_samples += len(inputs)
        species_loss_sum += species_loss
        total_loss = species_loss

        # Backpropagation and Parameter Update
        total_loss.backward()
        optimizer_spieces.step()

        # Calculate Species accuracy
        _, predicted_species = torch.max(species_output, 1)
        correct_species_train += (predicted_species == species_labels).sum().item()
        total_train += species_labels.size(0)

    avg_species_loss_sum = species_loss_sum / total_samples

    train_loss_list.append(avg_species_loss_sum)

    tqdm.write(f'\nloss_species: {avg_species_loss_sum}')

    train_species_accuracy = correct_species_train / total_train

    # Evaluate in the Validation set
    catdog_model.eval()
    with torch.no_grad():

        correct_species_val = 0
        total_val = 0

        val_loss_species = 0.0
        total_samples = 0

        for inputs, _, species_labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                species_labels = species_labels.cuda()
            species_output = catdog_model(inputs)

            # Move the output in the same device of gender_labels
            species_output = species_output.to(species_labels.device)
            species_labels = species_labels.long()

            val_loss_species += criterion_species(species_output, species_labels).item()
            total_samples += len(inputs)

            # Calculate Species accuracy
            _, predicted_species = torch.max(species_output, 1)
            correct_species_val += (predicted_species == species_labels).sum().item()
            total_val += species_labels.size(0)

    avg_val_loss_species = val_loss_species / total_samples
    total_val_loss = avg_val_loss_species

    val_loss_list.append(avg_val_loss_species)

    val_species_accuracy = correct_species_val / total_val

    train_species_accuracy_list.append(train_species_accuracy)
    val_species_accuracy_list.append(val_species_accuracy)

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    tqdm.write(f'\nEpoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_species_loss_sum:.4f}, '
          f'val Loss: {total_val_loss:.4f}, '
          f'No Improvement Count: {no_improvement_count}')

    tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Species Accuracy: {train_species_accuracy:.4f}, val Species Accuracy: {val_species_accuracy:.4f}')

    if no_improvement_count >= patience:
        print(f'Early stopping after {patience} epochs with no improvement.')
        break

end = time.time()
total_time = end - begin
print(f'Total Training Time: {total_time}')

torch.save(catdog_model.state_dict(), 'catdog_model.pth')

print(f'Total Training Samples: {total_train},  Total Validation Samples: {total_val}')

species_train_loss_list = train_loss_list[:]
species_val_loss_list = val_loss_list[:]

species_train_loss_list_cpu = [loss.detach().cpu().numpy() for loss in species_train_loss_list]

# ----------- TESTING CatsDogsClassifier DATASET ----------- #
catdog_model.eval()
with torch.no_grad():
    correct_species_test = 0
    total_test = 0

    test_loss_species = 0.0
    total_test_samples = 0

    all_predicted_species = []
    all_true_species = []


    for idx, (inputs, _, species_labels) in tqdm(enumerate(test_loader), desc="Test_loader"):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            species_labels = species_labels.cuda()
        species_output = catdog_model(inputs)

        # Move the output in the same device of gender_labels
        species_output = species_output.to(species_labels.device)
        species_labels = species_labels.long()

        test_loss_species += criterion_species(species_output, species_labels).item()
        total_test_samples += len(inputs)

        # Calculate gender accuracy
        _, predicted_species = torch.max(species_output, 1)
        correct_species_test += (predicted_species == species_labels).sum().item()
        total_test += species_labels.size(0)

        all_predicted_species.extend(predicted_species.cpu().numpy())
        all_true_species.extend(species_labels.cpu().numpy())


    test_species_accuracy = correct_species_test / total_test

    avg_test_loss_species = test_loss_species / total_test_samples

    tqdm.write(f'Test Loss: {avg_test_loss_species:.4f}')

    tqdm.write(f'Test CatsDogsClassifier Accuracy: {test_species_accuracy:.4f}')

    # Plot the confusion matrix
    confusion_matrix_1 = pd.crosstab(np.array(all_true_species), np.array(all_predicted_species), rownames=['Actual'],
                                   colnames=['Predicted'])

# ---------- GENERATING INPUTS FOR SECOND MODEL ---------- #

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 128

base_dataset = BreedDataset(annotations, transform=transform)
base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)

catdog_model.eval()
with torch.no_grad():
    all_predicted_species = []
    all_true_species = []

    for idx, (inputs, _, _) in tqdm(enumerate(base_loader), desc="first_hierarchi_loader"):
        if torch.cuda.is_available():
            catdog_model = catdog_model.cuda()
            inputs = inputs.cuda()
        species_output = catdog_model(inputs)

        _, predicted_species = torch.max(species_output, 1)
        all_predicted_species.extend(predicted_species.cpu().numpy())

print(all_predicted_species)
# --------------- SECOND MODEL --------------#
print(annotations.head())
annotations['SPECIES'] = all_predicted_species
print(annotations.head())
proj_df, test_df = train_test_split(annotations, test_size=0.3, random_state=42)

train_df, val_df = train_test_split(proj_df, test_size=0.3, stratify=proj_df['classname'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

base_train_dataset = BreedDataset(train_df, transform=transform)
aug_train_dataset = BreedDataset(train_df, transform=augmentation_transform)
train_dataset = ConcatDataset([base_train_dataset, aug_train_dataset])

val_dataset = BreedDataset(val_df, transform=transform)
test_dataset = BreedDataset(test_df, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class BreedClassifier(nn.Module):
    def __init__(self):
        super(BreedClassifier, self).__init__()

        # CNN architecture
        # CNN architecture
        self.breed_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # BREED Branch
        self.breed_fc = nn.Sequential(
            # nn.Linear(65536, 1024),
            nn.Linear(16385, 1024),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        img, class_feature = x
        # Forward pass through the CNN
        cnn_output = self.breed_cnn(img)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # Flatten the CNN output
        # Ensure class_feature has the correct shape
        class_feature = class_feature.view(class_feature.size(0), 1)  # Assuming class_feature is a single value
        # Concatenate CNN output with class_feature
        combined_input = torch.cat((cnn_output, class_feature), dim=1)
        # Forward pass through the fully connected layers
        breed_output = self.breed_fc(combined_input)

        return breed_output

    def get_conv_layers(self):
        return [idx for idx, layer in enumerate(self.modules()) if isinstance(layer, nn.Conv2d)]

    def generate_heatmaps(self, sample_images):
        # Set the model to evaluation mode
        self.cuda().eval()
        sample_images , species_labels =  sample_images
        activations = []
        hooks = []
        for layer_idx in self.get_conv_layers():
            conv_layer = list(self.modules())[layer_idx]
            hook = conv_layer.register_forward_hook(self.get_activation_hook(activations))
            hooks.append(hook)

        with torch.no_grad():
            self((sample_images, species_labels))

        for hook in hooks:
            hook.remove()

        # Calculate mean activations and create heatmaps
        mean_activations = [activation.mean(dim=1, keepdim=True) for activation in activations]
        heatmaps = [torch.nn.functional.interpolate(mean_activation, size=sample_images.shape[2:], mode='bilinear',
                                                    align_corners=False) for mean_activation in mean_activations]

        return heatmaps

    def get_activation_hook(self, activations):
        def hook(module, input, output):
            activations.append(output.cpu())

        return hook


# ----------- TRAINING AND VALIDATION BreedClassifier MODELS ----------- #
breed_model = BreedClassifier()
criterion_breed = nn.CrossEntropyLoss()

optimizer_breed = optim.Adam(breed_model.parameters(), lr=0.0001, weight_decay=0.001)
# model.load_state_dict(torch.load('breed_model_1.pth', map_location=torch.device('gpu')))


# Max num of epochs
num_epochs = 50

# Loss lists
train_loss_list = []
val_loss_list = []

# Accuracy lists
train_breed_accuracy_list = []
val_breed_accuracy_list = []

if torch.cuda.is_available():
    breed_model_1 = breed_model.cuda()

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
no_improvement_count = 0

# Training the model
begin = time.time()
for epoch in tqdm(range(num_epochs), desc="Epochs"):

    correct_breed_train = 0
    total_train = 0
    breed_loss_sum = 0
    total_samples = 0

    breed_model.train()

    for inputs, breed_labels, species_labels in train_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
            species_labels = species_labels.cuda()
        optimizer_breed.zero_grad()
        breed_output = breed_model((inputs, species_labels))

        # Move the output in the same device of gender_labels
        breed_output = breed_output.to(breed_labels.device)
        breed_labels = breed_labels.long()

        # Calculate loss
        breed_loss = criterion_breed(breed_output, breed_labels)
        total_samples += len(inputs)
        breed_loss_sum += breed_loss
        total_loss = breed_loss

        # Backpropagation and Parameter Update
        total_loss.backward()
        optimizer_breed.step()

        # Calculate Breed accuracy
        _, predicted_breed = torch.max(breed_output, 1)
        correct_breed_train += (predicted_breed == breed_labels).sum().item()
        total_train += breed_labels.size(0)

    avg_breed_loss_sum = breed_loss_sum / total_samples

    train_loss_list.append(avg_breed_loss_sum)

    tqdm.write(f'\nloss_breed: {avg_breed_loss_sum}')

    train_breed_accuracy = correct_breed_train / total_train

    # Evaluate in the Validation set
    breed_model.eval()
    with torch.no_grad():

        correct_breed_val = 0
        total_val = 0

        val_loss_breed = 0.0
        total_samples = 0

        for inputs, breed_labels, species_labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                breed_labels = breed_labels.cuda()
                species_labels = species_labels.cuda()
            breed_output = breed_model((inputs, species_labels))

            # Move the output in the same device of gender_labels
            breed_output = breed_output.to(breed_labels.device)
            breed_labels = breed_labels.long()

            val_loss_breed += criterion_breed(breed_output, breed_labels).item()
            total_samples += len(inputs)

            # Calculate Breed accuracy
            _, predicted_breed = torch.max(breed_output, 1)
            correct_breed_val += (predicted_breed == breed_labels).sum().item()
            total_val += breed_labels.size(0)

    avg_val_loss_breed = val_loss_breed / total_samples
    total_val_loss = avg_val_loss_breed

    val_loss_list.append(avg_val_loss_breed)

    val_breed_accuracy = correct_breed_val / total_val

    train_breed_accuracy_list.append(train_breed_accuracy)
    val_breed_accuracy_list.append(val_breed_accuracy)

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    tqdm.write(f'\nEpoch [{epoch + 1}/{num_epochs}], '
               f'Train Loss: {avg_breed_loss_sum:.4f}, '
               f'val Loss: {total_val_loss:.4f}, '
               f'No Improvement Count: {no_improvement_count}')

    tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], '
               f'Train Breed Accuracy: {train_breed_accuracy:.4f}, val Breed Accuracy: {val_breed_accuracy:.4f}')

    if no_improvement_count >= patience:
        print(f'Early stopping after {patience} epochs with no improvement.')
        break

end = time.time()
total_time = end - begin
print(f'Total Training Time: {total_time}')

torch.save(breed_model.state_dict(), 'breed_model_second.pth')

print(f'Total Training Samples: {total_train},  Total Validation Samples: {total_val}')

breed_train_loss_list = train_loss_list[:]
breed_val_loss_list = val_loss_list[:]

breed_train_loss_list_cpu = [loss.detach().cpu().numpy() for loss in breed_train_loss_list]

# ----------- TESTING BreedClassifier DATASET ----------- #
breed_model.eval()
with torch.no_grad():
    correct_breed_test = 0
    total_test = 0

    test_loss_breed = 0.0
    total_test_samples = 0

    all_predicted_breeds = []
    all_true_breeds = []

    for idx, (inputs, breed_labels, species_labels) in tqdm(enumerate(test_loader), desc="Test_loader"):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
            species_labels = species_labels.cuda()
        breed_output = breed_model((inputs, species_labels))

        # Move the output in the same device of gender_labels
        breed_output = breed_output.to(breed_labels.device)
        breed_labels = breed_labels.long()

        test_loss_breed += criterion_breed(breed_output, breed_labels).item()
        total_test_samples += len(inputs)

        # Calculate gender accuracy
        _, predicted_breed = torch.max(breed_output, 1)
        correct_breed_test += (predicted_breed == breed_labels).sum().item()
        total_test += breed_labels.size(0)

        all_predicted_breeds.extend(predicted_breed.cpu().numpy())
        all_true_breeds.extend(breed_labels.cpu().numpy())

    test_breed_accuracy = correct_breed_test / total_test

    avg_test_loss_breed = test_loss_breed / total_test_samples

    tqdm.write(f'Test Loss: {avg_test_loss_breed:.4f}')

    tqdm.write(f'Test BreedClassifier Accuracy: {test_breed_accuracy:.4f}')

    # Plot the confusion matrix
    confusion_matrix = pd.crosstab(np.array(all_true_breeds), np.array(all_predicted_breeds), rownames=['Actual'],
                                   colnames=['Predicted'])

# ----------- CUSTOM MODELS HEATMAPS AND LOSS ----------- #

# BreedClassifier_1 ------------------------ #
plt.figure(figsize=(8, 6), dpi=80)

# Plot loss trend
plt.plot(breed_train_loss_list_cpu, label='Train Loss')
plt.plot(breed_val_loss_list, 'r', label='Validation Loss',
         alpha=0.3)  # Usiamo il colore rosso per la perdita di addestramento
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("BreedClassifier Loss trend")
plt.show()

validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
sample_images, _, sample_species = next(iter(validation_dataloader))

# Generate heatmaps
heatmaps_1 = breed_model.generate_heatmaps((sample_images.cuda(), sample_species.cuda()))

# Plot the heatmaps
fig, axs = plt.subplots(2, int(len(heatmaps_1) / 2), figsize=(8, 8))

# Plot the heatmaps
for i, heatmap in enumerate(heatmaps_1):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(heatmap.squeeze().numpy(), cmap='viridis')
    axs[row, col].set_title(f'BreedClassifier G Conv Layer {i + 1} Activation')
    axs[row, col].axis('off')

plt.show()

# ----------- CUSTOM MODELS HEATMAPS ----------- #

# CatsDogsClassifier ------------------------ #
plt.figure(figsize=(8, 6), dpi=80)

# Plot loss trend
plt.plot(species_train_loss_list_cpu, label='Train Loss')
plt.plot(species_val_loss_list, 'r', label='Validation Loss',
         alpha=0.3)  # Usiamo il colore rosso per la perdita di addestramento
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("CatsDogsClassifier Loss trend")
plt.show()

validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
sample_images, _, _= next(iter(validation_dataloader))

# Generate heatmaps
heatmaps = catdog_model.generate_heatmaps(sample_images.cuda())

# Plot the heatmaps
fig, axs = plt.subplots(2, int(len(heatmaps) / 2), figsize=(8, 8))

# Plot the heatmaps
for i, heatmap in enumerate(heatmaps):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(heatmap.squeeze().numpy(), cmap='viridis')
    axs[row, col].set_title(f'CatsDogsClassifier G Conv Layer {i + 1} Activation')
    axs[row, col].axis('off')

plt.show()

# ----------- RESULTS OF ALL TESTED MODELS ----------- #

# Results of all models

print(f'Test CatsDogsClassifier Accuracy: {test_species_accuracy:.4f}')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_1, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Species Classification Accuracy({test_species_accuracy:.4f})')
plt.show()

# ----------- RESULTS OF ALL TESTED MODELS ----------- #

# Results of all models

print(f'Test BreedClassifier_1 Accuracy: {test_breed_accuracy:.4f}')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Breed Classification Accuracy({test_breed_accuracy:.4f})')
plt.show()
