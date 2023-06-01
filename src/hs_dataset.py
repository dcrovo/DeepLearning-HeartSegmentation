"""
Developed by: Daniel Crovo
Dataset class definition

"""
from torch.utils.data import Dataset
import os
from PIL import Image 
from pydicom import dcmread
from dicom_utils import apply_windowing
import numpy as np

class HSDataset(Dataset): 
    def __init__(self, image_dir, mask_dir, transform = None, w_level=35, w_width=350, normalized =False) -> None:
        """Initialises the Heart Segmentation Dataset class. The dataset asumes all the dicom files are in one single folder aswell as the masks
            both the images (dicom files) and masks should have the same name for each slice
        Args:
            image_dir (string): Path to the dicom directory
            mask_dir (string): Path to the masks directory 
            transform (_type_, optional):Aplied transformations Defaults to None.
            w_level (int, optional): The window level (center). Defaults to 35.
            w_width (int, optional): The width of the window Defaults to 400.
            normalized (bool, optional): wheter to normalized the windowed array after applying windoing
        """
        super().__init__()
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.w_level = w_level
        self.w_width = w_width
        self.normalized = normalized
        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        ds = dcmread(img_path)
        dicom_img = apply_windowing(ds = ds,window_center = self.w_level, 
                                    window_width =self.w_width, normalized = self.normalized)
        image = Image.fromarray(dicom_img).convert('RGB')
        image = np.array(image)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask[mask == 255.0] = 1.0


        if self.transform is not None:
            transformations = self.transform(image = image, mask = mask)
            image = transformations['image']
            mask = transformations['mask']
            
        return image, mask

