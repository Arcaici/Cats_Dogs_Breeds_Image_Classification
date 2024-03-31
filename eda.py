import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
import re

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

from Dataset import CatsDogsDataset


def parse_breed_from_xmls(xml_dir):
    """
    Parses the XML annotation file and returns a list of all images size.

    Args:
        xml_dir (str): Path to the XML dir containing annotation files.

    Returns:
        breed  (list): one list containing all the animal breed
    """
    breed = []
    pattern = r'_[0-9]+\.\w+$'
    xml_paths = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_paths = [os.path.join(xml_dir, f) for f in xml_paths]
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        filename = root.find("filename").text
        breed.append(re.sub(pattern, '', filename))

    breed = dict(Counter(breed))

    return breed

def parse_types_from_xmls(xml_dir):
    """
    Parses the XML annotation file and returns a list of all images size.

    Args:
        xml_dir (str): Path to the XML dir containing annotation files.

    Returns:
        type  (list): one list containing all the animal type (cat, dog).
    """
    types = []
    xml_paths = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_paths = [os.path.join(xml_dir, f) for f in xml_paths]
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        filename = root.find("filename").text
        if filename[0].isupper():
            types.append("Cat")
        else:
            types.append("Dog")

    types = dict(Counter(types))


    return types

def parse_size_from_xmls(xml_dir):
    """
    Parses the XML annotation file and returns a list of all images size.

    Args:
        xml_dir (str): Path to the XML dir containing annotation files.

    Returns:
        sizes  (list): one list containing all the images size.
    """
    sizes = []
    xml_paths = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_paths = [os.path.join(xml_dir, f) for f in xml_paths]
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find("size")
        width = size.find("width").text
        height = size.find("height").text
        sizes.append(width+"x"+height)

    sizes = dict(Counter(sizes))

    return sizes

def parse_pose_from_xmls(xml_dir):
    """
    Parses the XML annotation file and returns a list of all animal pose.

    Args:
        xml_dir (str): Path to the XML dir containing annotation files.

    Returns:
        poses  (list): one list containing all the animal pose.
    """
    poses = []
    xml_paths = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_paths = [os.path.join(xml_dir, f) for f in xml_paths]
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        object = root.find("object")
        poses.append(object.find("pose").text)

    poses = dict(Counter(poses))

    return poses

def count_imgages_and_xmls_files(dir_images, dir_annotation):
    """
        Count the XML annotation file and Images file returns a list of all number of files.

        Args:
            dir_images (str): Path to the dir containing all images
            dir_annotation (str): Path to the XML dir containing annotation files.

        Returns:
            files  (list): one list containing all images number and annotations numbers.
        """
    number_of_images = len(os.listdir(dir_images))
    number_of_xml = len(os.listdir(dir_annotation))

    files = {"images": number_of_images, "xmls": number_of_xml}
    return files
def plot_dict(dict, title="Dict plots"):
    """
        Plot a bar chart that show the first 20 resolution after sorting values in descending order.

            Args:
                dict    dict(str)(int): dict containing occurrences and units to plot
                title            (str): plot title

    """
    if len(dict.keys()) > 20:
        sorted_data = sorted(dict.items(), key=lambda x: x[1], reverse=True)[:20]
        title += f" first 20 keys of {len(dict.keys())}"
    else:
        sorted_data = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    # Extract keys and values from the sorted data
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    # Plotting the data
    plt.figure(figsize=(8, 6))
    bars = plt.bar(keys, values)
    plt.xlabel('Keys')
    plt.ylabel('Occurrences')
    plt.title(title)

    # Adding count labels above each bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')


    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_random_images(all_images, num_images=4):
    """
        Display a certain number of images

            Args:
                all_images    (Datset): dataset of images
                num_images       (int): number of images to plot

    """
    plt.figure(figsize=(8, 12))
    for i in range(num_images):
        idx = random.randint(0, len(all_images) - 1)
        image, classID = all_images[idx]
        classname = all_images.annotations['classname'].iloc[idx]
        plt.subplot(4, 3, i + 1)
        plt.imshow(image)
        plt.title(classname)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# # ---------------- USING XMLS FILES FOR EDA ---------------- #
# dir_annotation = "./annotations/xmls"
# dir_images = "./images"
# sizes = parse_size_from_xmls(dir_annotation)
# poses = parse_pose_from_xmls(dir_annotation)
# types = parse_types_from_xmls(dir_annotation)
# breeds = parse_breed_from_xmls(dir_annotation)
# images_and_xmls = count_imgages_and_xmls_files(dir_images, dir_annotation)
#
# print(sizes)
# print(breeds)
# plot_dict(sizes, title="All image sizes")
# plot_dict(breeds, title="All image breeds")
# plot_dict(poses, title="All image poses")
# plot_dict(types, title="All image animal types")
# plot_dict(images_and_xmls, title="All files")
# print("Mean images units per breed: ", np.mean(list(breeds.values())))
# print("Standard Deviation images units per breed: ", np.std(list(breeds.values())))
#
# # the number of xml files are less than images files. I'll use only list.txt has annotation file.

# ---------------- USING list.txt FILE FOR EDA ---------------- #

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

# ---------- USING DATASET CLASS FOR EDA ---------- #
all_images = CatsDogsDataset(annotations)
plot_random_images(all_images, num_images=12)

# Looking for images size
dataset_size = len(all_images)
print(dataset_size)
image_sizes = []
for i in range(dataset_size):
    img, _ = all_images[i]
    image_sizes.append(img.size)

unique_count = len(set(image_sizes))
print("Number of unique elements:", unique_count)
x_values, y_values = zip(*image_sizes)

num_bins = 50
hist_x, bins_x = np.histogram(x_values, bins=num_bins, range=(np.min(x_values), np.max(x_values)))
hist_y, bins_y = np.histogram(y_values, bins=num_bins, range=(np.min(y_values), np.max(y_values)))

plt.bar(bins_x[:-1], hist_x, width=(np.max(x_values) - np.min(x_values))/num_bins, align='edge')
plt.xlabel('X value')
plt.ylabel('Frequency')
plt.title(f'X values Histogram with {num_bins} Bins')
plt.grid(True)
plt.show()

plt.bar(bins_y[:-1], hist_y, width=(np.max(y_values) - np.min(y_values))/num_bins, align='edge')
plt.xlabel('Y value')
plt.ylabel('Frequency')
plt.title(f'Y values Histogram with {num_bins} Bins')
plt.grid(True)
plt.show()

filtered_data = [(x, y) for x, y in image_sizes if x < 1000 and y < 1000]

x_values, y_values = zip(*filtered_data)
plt.hist2d(x_values, y_values, bins=(50, 50), cmap='viridis', cmin = 1)

plt.xlim(0, 1000)
plt.ylim(0, 1000)

# Add color bar for reference
cbar = plt.colorbar()
cbar.set_label('Frequency')

# Add labels and title
plt.xlabel('horizontal')
plt.ylabel('vertical')
plt.title('2D Histogram for resolutions')

# Show the plot
plt.show()

hist, x_edges, y_edges, _ = plt.hist2d(x_values, y_values, bins=(30, 30), cmap='inferno', cmin = 1, vmax=30)

plt.xlim(0, 1000)
plt.ylim(0, 1000)

# Add color bar for reference
cbar = plt.colorbar()
cbar.set_label('Frequency')

cbar.set_ticks([0, 2, 4, 6, 10, 20, 30])

plt.xlabel('horizontal')
plt.ylabel('vertical')
plt.title('2D Histogram for rare resolutions')
plt.show()

plt.hist(annotations['classname'], bins=37, edgecolor='black',rwidth=0.5)
plt.xticks(rotation='vertical')
plt.xlabel('breed indexes')
plt.ylabel('Frequency')
plt.title('Frequency of different breeds in dataset')
plt.show()

# ----------------- SPLITTING DATASET ----------------- #
proj_df, test_df = train_test_split(annotations, test_size=0.2, random_state=42, shuffle=True)
train_df, val_df = train_test_split(proj_df,stratify=annotations["classname"], test_size=0.3, random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

