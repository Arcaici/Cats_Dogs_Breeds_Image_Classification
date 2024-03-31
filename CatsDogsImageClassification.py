import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from Dataset import CatsDogsDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchinfo import summary

# ----------- LOADING DATASET ----------- #

# Loading list.txt
dir_annotation = "./annotations/list.txt"
annotations = pd.read_csv(dir_annotation, skiprows=6, header=None, names=['#Image CLASS-ID SPECIES BREED ID'])
print(annotations.head())

# preprocessing information
annotations[['CLASS-ID','SPECIES','BREED','ID']] = annotations['#Image CLASS-ID SPECIES BREED ID'].str.split(expand=True)
annotations = annotations.drop('#Image CLASS-ID SPECIES BREED ID',axis=1)
annotations = annotations.rename(columns={"CLASS-ID": "image", "SPECIES": "CLASS-ID", 'BREED' : "SPECIES", "ID":"BREED ID"})
annotations[["CLASS-ID","SPECIES","BREED ID"]] = annotations[["CLASS-ID","SPECIES","BREED ID"]].astype(int)
print(annotations.head())

# adding the extension to image, so it can be used to access the real image
annotations['image'] = annotations['image'].apply(lambda x : str(x)+'.jpg')
annotations = annotations.reset_index()
annotations = annotations.drop('index',axis=1)

#Extracting the classname/breed of the animal
annotations['classname'] = annotations['image'].apply(lambda x: str(x)[:str(x).rindex('_')])

# Adding information about cat or dog based on the 'Species' column to the 'classname' column
annotations['classname'] = annotations.apply(lambda row: f"{('dog' if row['SPECIES'] == 2 else 'cat')}_{row['classname']}", axis=1)
print(annotations.head(5))

proj_df, test_df = train_test_split(annotations, test_size=0.3, random_state=42)

train_df, val_df = train_test_split(proj_df, test_size=0.3, stratify=proj_df['classname'], random_state=42)

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

base_train_dataset = CatsDogsDataset(train_df, transform=transform)
aug_train_dataset = CatsDogsDataset(train_df,  transform=augmentation_transform)
train_dataset = ConcatDataset([base_train_dataset, aug_train_dataset])

val_dataset = CatsDogsDataset(val_df, transform=transform)
test_dataset = CatsDogsDataset(test_df, transform=transform)

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
class BreedClassifier_5(nn.Module):
    def __init__(self):
        super(BreedClassifier_5, self).__init__()

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
            nn.Linear(16384, 1024),
            # nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        cnn_output = self.breed_cnn(x)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)
        breed_output = self.breed_fc(cnn_output)

        return breed_output

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

class BreedClassifier_6(nn.Module):
    def __init__(self):
        super(BreedClassifier_6, self).__init__()

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
            nn.Linear(16384, 1024),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        cnn_output = self.breed_cnn(x)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)
        breed_output = self.breed_fc(cnn_output)

        return breed_output

    def get_conv_layers(self):
        return [idx for idx, layer in enumerate(self.modules()) if isinstance(layer, nn.Conv2d)]

    def generate_heatmaps_2(self, sample_images):
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

        # Calculate mean activations and create heatmaps_2
        mean_activations = [activation.mean(dim=1, keepdim=True) for activation in activations]
        heatmaps_2 = [torch.nn.functional.interpolate(mean_activation, size=sample_images.shape[2:], mode='bilinear',
                                                    align_corners=False) for mean_activation in mean_activations]

        return heatmaps_2

    def get_activation_hook(self, activations):
        def hook(module, input, output):
            activations.append(output.cpu())

        return hook

# ----------- TRAINING AND VALIDATION BreedClassifier_1 MODELS ----------- #
# Create model
breed_model_1 = BreedClassifier_5()

summary_str = summary(breed_model_1, input_size= (batch_size, 3, 128, 128))

criterion_breed = nn.CrossEntropyLoss()

optimizer_breed = optim.Adam(breed_model_1.parameters(), lr=0.0001, weight_decay=0.001)
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
    breed_model_1 = breed_model_1.cuda()

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

    breed_model_1.train()

    for inputs, breed_labels in train_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
        optimizer_breed.zero_grad()
        breed_output = breed_model_1(inputs)

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
    breed_model_1.eval()
    with torch.no_grad():

        correct_breed_val = 0
        total_val = 0

        val_loss_breed = 0.0
        total_samples = 0

        for inputs, breed_labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                breed_labels = breed_labels.cuda()
            breed_output = breed_model_1(inputs)

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

torch.save(breed_model_1.state_dict(), 'breed_model_1.pth')

print(f'Total Training Samples: {total_train},  Total Validation Samples: {total_val}')

breed_train_loss_list_1 = train_loss_list[:]
breed_val_loss_list_1 = val_loss_list[:]

breed_train_loss_list_cpu_1 = [loss.detach().cpu().numpy() for loss in breed_train_loss_list_1]

# ----------- TRAINING AND VALIDATION BreedClassifier_2 MODELS ----------- #
# Create model
breed_model_2 = BreedClassifier_6()

summary_str = summary(breed_model_2, input_size= (batch_size, 3, 128, 128))

criterion_breed = nn.CrossEntropyLoss()
optimizer_breed = optim.Adam(breed_model_2.parameters(), lr=0.0001, weight_decay=0.001)
# model.load_state_dict(torch.load('breed_model_2.pth', map_location=torch.device('gpu')))


# Max num of epochs
num_epochs = 50

# Loss lists
train_loss_list = []
val_loss_list = []

# Accuracy lists
train_breed_accuracy_list = []
val_breed_accuracy_list = []

if torch.cuda.is_available():
    breed_model_2 = breed_model_2.cuda()

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

    breed_model_2.train()

    for inputs, breed_labels in train_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
        optimizer_breed.zero_grad()
        breed_output = breed_model_2(inputs)

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
    breed_model_2.eval()
    with torch.no_grad():

        correct_breed_val = 0
        total_val = 0

        val_loss_breed = 0.0
        total_samples = 0

        for inputs, breed_labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                breed_labels = breed_labels.cuda()
            breed_output = breed_model_2(inputs)

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

torch.save(breed_model_2.state_dict(), 'breed_model_2.pth')

print(f'Total Training Samples: {total_train},  Total Validation Samples: {total_val}')

breed_train_loss_list_2 = train_loss_list[:]
breed_val_loss_list_2 = val_loss_list[:]

breed_train_loss_list_cpu_2 = [loss.detach().cpu().numpy() for loss in breed_train_loss_list_2]


# ----------- TESTING BreedClassifier_1 DATASET ----------- #
breed_model_1.eval()
with torch.no_grad():
    correct_breed_test = 0
    total_test = 0

    test_loss_breed = 0.0
    total_test_samples = 0

    all_predicted_breeds = []
    all_true_breeds = []


    for idx, (inputs, breed_labels) in tqdm(enumerate(test_loader), desc="Test_loader"):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
        breed_output = breed_model_1(inputs)

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


    test_breed_accuracy_1 = correct_breed_test / total_test

    avg_test_loss_breed = test_loss_breed / total_test_samples

    tqdm.write(f'Test Loss: {avg_test_loss_breed:.4f}')

    tqdm.write(f'Test BreedClassifier_1 Accuracy: {test_breed_accuracy_1:.4f}')

    # Plot the confusion matrix
    confusion_matrix_1 = pd.crosstab(np.array(all_true_breeds), np.array(all_predicted_breeds), rownames=['Actual'],
                                   colnames=['Predicted'])

# ----------- TESTING BreedClassifier_2 DATASET ----------- #
breed_model_2.eval()
with torch.no_grad():
    correct_breed_test = 0
    total_test = 0

    test_loss_breed = 0.0
    total_test_samples = 0

    all_predicted_breeds = []
    all_true_breeds = []


    for idx, (inputs, breed_labels) in tqdm(enumerate(test_loader), desc="Test_loader"):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            breed_labels = breed_labels.cuda()
        breed_output = breed_model_2(inputs)

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


    test_breed_accuracy_2 = correct_breed_test / total_test

    avg_test_loss_breed = test_loss_breed / total_test_samples

    tqdm.write(f'Test Loss: {avg_test_loss_breed:.4f}')

    tqdm.write(f'Test BreedClassifier_2 Accuracy: {test_breed_accuracy_2:.4f}')

    # Plot the confusion matrix
    confusion_matrix_2 = pd.crosstab(np.array(all_true_breeds), np.array(all_predicted_breeds), rownames=['Actual'],
                                   colnames=['Predicted'])


# ----------- TESTING model_pretrained_vgg16 DATASET ----------- #
# Load pre-trained VGG16 model
vgg_model = models.vgg16(pretrained=True)

# Freeze all the layers of the VGG16 model
for param in vgg_model.parameters():
    param.requires_grad = False

# Extract features from VGG16 model up to the specified layer
conv_features = vgg_model.features[:24]  # Assuming 'block4_pool' is at index 24

# Define additional layers
global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
fc1 = nn.Linear(512, 1024)
relu = nn.ReLU(inplace=True)
dropout = nn.Dropout(0.25)
fc2 = nn.Linear(1024, 37)

# Combine all layers
model_pretrained_vgg16 = nn.Sequential(conv_features, global_avg_pooling, nn.Flatten(), fc1, relu, dropout, fc2)

# Print the structure of the model
print(model_pretrained_vgg16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pretrained_vgg16 = model_pretrained_vgg16.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pretrained_vgg16.parameters(), lr=0.001)

# Define early stopping parameters
patience = 5
best_val_loss = float('inf')
no_improvement_count = 0
all_predicted_breeds = []
all_true_breeds = []
# Train the model
num_epochs = 50
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model_pretrained_vgg16.train()  # Set the model to train mode
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_pretrained_vgg16(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss for the epoch
    train_loss /= len(train_loader)

    # Validate the model
    model_pretrained_vgg16.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_pretrained_vgg16(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    val_loss /= len(val_loader)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_count = 0
        best_model_state_dict = model_pretrained_vgg16.state_dict()
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print(f'Early stopping after {patience} epochs without improvement.')
        break

torch.save(model_pretrained_vgg16.state_dict(), 'breed_model_vgg.pth')

# Test the model
model_pretrained_vgg16.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_pretrained_vgg16(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predicted_breeds.extend(predicted.cpu().numpy())
        all_true_breeds.extend(labels.cpu().numpy())

test_accuracy = correct / total
confusion_matrix_vgg = pd.crosstab(np.array(all_true_breeds), np.array(all_predicted_breeds), rownames=['Actual'],
                                   colnames=['Predicted'])

# ----------- CUSTOM MODELS HEATMAPS ----------- #

# BreedClassifier_1 ------------------------ #
plt.figure(figsize=(8, 6), dpi=80)

# Plot loss trend
plt.plot(breed_train_loss_list_cpu_1, label='Train Loss')
plt.plot(breed_val_loss_list_1, 'r', label='Validation Loss',
         alpha=0.3)  # Usiamo il colore rosso per la perdita di addestramento
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("BreedClassifier_1 Loss trend")
plt.show()

validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
sample_images, _= next(iter(validation_dataloader))

# Generate heatmaps
heatmaps_1 = breed_model_1.generate_heatmaps(sample_images.cuda())
# Plot the heatmaps
fig, axs = plt.subplots(2, int(len(heatmaps_1) / 2), figsize=(8, 8))

# Plot the heatmaps
for i, heatmap in enumerate(heatmaps_1):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(heatmap.squeeze().numpy(), cmap='viridis')
    axs[row, col].set_title(f'BreedClassifier_1 G Conv Layer {i + 1} Activation')
    axs[row, col].axis('off')

plt.show()

# BreedClassifier_2 ------------------------ #

plt.figure(figsize=(8, 6), dpi=80)

# Plot loss trend
plt.plot(breed_train_loss_list_cpu_2, label='Train Loss')
plt.plot(breed_val_loss_list_2, 'r', label='Validation Loss',
         alpha=0.3)  # Usiamo il colore rosso per la perdita di addestramento
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("BreedClassifier_2 Loss trend")
plt.show()

validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
sample_images, _= next(iter(validation_dataloader))

# Generate heatmaps_2
heatmaps_2 = breed_model_2.generate_heatmaps_2(sample_images.cuda())
# Plot the heatmaps_2
fig, axs = plt.subplots(2, int(len(heatmaps_2) / 2), figsize=(8, 8))

# Plot the heatmaps_2
for i, heatmap in enumerate(heatmaps_2):
    try:
        row = i // 2
        col = i % 2
        axs[row, col].imshow(heatmap.squeeze().numpy(), cmap='viridis')
        axs[row, col].set_title(f'BreedClassifier_2 G Conv Layer {i + 1} Activation')
        axs[row, col].axis('off')
    except IndexError:
        print(f"Index {i} is out of bounds for subplot array.")
plt.show()



# ----------- RESULTS OF ALL TESTED MODELS ----------- #

# Results of all models

print(f'Test BreedClassifier_1 Accuracy: {test_breed_accuracy_1:.4f}')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_1, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Breed Classification 1 Accuracy({test_breed_accuracy_1:.4f})')
plt.show()

print(f'Test BreedClassifier_2 Accuracy: {test_breed_accuracy_2:.4f}')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_2, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Breed Classification 2 Accuracy({test_breed_accuracy_2:.4f})')
plt.show()

print(f"Test Accuracy: {test_accuracy:.2f}%")
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_vgg, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Breed Classification vgg Accuracy({test_accuracy:.2f})%')
plt.show()