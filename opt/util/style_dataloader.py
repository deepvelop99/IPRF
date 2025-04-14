import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class StyledImageDataset(Dataset):
    def __init__(self, image_folder_a, image_folder_b, transform=None):
        self.image_folder_a = image_folder_a
        self.image_folder_b = image_folder_b
        self.transform = transform
        self.image_files_a = [os.path.join(image_folder_a, f) for f in os.listdir(image_folder_a) if os.path.isfile(os.path.join(image_folder_a, f))]
        self.image_files_b = [os.path.join(image_folder_b, f) for f in os.listdir(image_folder_b) if os.path.isfile(os.path.join(image_folder_b, f))]
        assert len(self.image_files_a) == len(self.image_files_b), "The two folders must have the same number of images"

    def __len__(self):
        return len(self.image_files_a)

    def __getitem__(self, idx):
        try:
            image_a = Image.open(self.image_files_a[idx]).convert('RGB')
            image_b = Image.open(self.image_files_b[idx]).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images: {e}")

        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        # 이미지 크기 확인
        if image_a.size != image_b.size:
            raise ValueError(f"Image sizes do not match: {image_a.size} vs {image_b.size}")

        return image_a, image_b

# 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])