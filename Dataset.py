from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.folder_patch = './images/'
        self.annotations = file_list
        self.transform = transform
        self.filelength = len(file_list)

    def __len__(self):
        return self.filelength

    def __getitem__(self, idx):
        classID = self.annotations['CLASS-ID'].iloc[idx]
        img_path = self.annotations['image'].iloc[idx]
        img_path = self.folder_patch + img_path
        img = Image.open(img_path)

        img = img.convert('RGB')
        if self.transform is not None:
            try:
                img = self.transform(img)
            except RuntimeError as e:
                print(f"Exception: {e}")
                print("Shape before normalization:", img.size)
                print(img_path)
                tot = transforms.ToTensor()
                img_tensor = tot(img)
                print("Input Tensor Shape:", img_tensor.shape)
                print("Input Tensor Values:", img_tensor)

        return img, classID - 1