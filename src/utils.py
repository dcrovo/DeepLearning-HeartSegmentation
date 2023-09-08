"""
Developed by: Daniel Crovo

"""
import numpy as np
import csv
from hs_dataset import HSDataset
from torch.utils.data import DataLoader
import torch
import os
import torchvision
from torchmetrics.classification import BinaryJaccardIndex
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
from PIL import Image

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, 
                val_transform, num_workers, w_level, w_width, pin_memory = True, normalize=False):
    """_summary_

    Args:
        train_dir (_type_): _description_
        train_maskdir (_type_): _description_
        val_dir (_type_): _description_
        val_maskdir (_type_): _description_
        batch_size (_type_): _description_
        train_transform (_type_): _description_
        val_transform (_type_): _description_
        num_workers (_type_): _description_
        pin_memory (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    train_img_mask = HSDataset(image_dir = train_dir, 
                               mask_dir = train_maskdir, transform = train_transform, 
                               normalized=normalize, w_level=w_level, w_width=w_width)
    train_loader = DataLoader(train_img_mask, batch_size = batch_size, 
                              num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
    val_img_mask = HSDataset(image_dir = val_dir, mask_dir = val_maskdir, 
                             transform = val_transform, normalized=normalize,
                             w_level=w_level, w_width=w_width,)
    val_loader = DataLoader(val_img_mask, batch_size = batch_size, 
                            num_workers = num_workers, pin_memory = pin_memory, shuffle = False)
    
    return train_loader, val_loader


def load_checkpoint(checkpoint, model): 
    try: 
        model.load_state_dict(checkpoint['state_dict'])
        print('\nCheckpoint importado exitosamente.')
    except: 
        print('Error en la importación del Checkpoint.')
    
def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'): 
    try: 
        torch.save(state, filename)
        print('\nCheckpoint almacenado exitosamente.')
    except: 
        print('Error en la importación del Checkpoint.')

def compute_jaccard_index(preds, targets):
    intersection = torch.logical_and(preds, targets).sum()
    union = torch.logical_or(preds, targets).sum()
    jaccard = intersection.item() / (union.item()+ 1e-10)
    return jaccard
def diceCoef(y_true, y_pred, smooth=1.):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * y_pred_f)
  dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
  return round(float(dice), 3)

def perf(loader, model, device): 
    dice_score = 0.0
    jaccard = 0.0

    model.eval()
    
    with torch.no_grad(): # deshabilitar el cálculo y almacenamiento de gradientes en el grafo computacional de PyTorch
        for x, y in loader: 
            x = x.to(device = device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            
            preds = (preds > 0.5).float()
            if((preds.sum() == 0) and (y.sum() == 0)):
                jaccard +=1
                dice_score +=1
            else:
                jaccard += compute_jaccard_index(preds, y)
                dice_score += (2*(preds*y).sum())/((preds + y).sum() + 1e-10)

    dice_s = dice_score/len(loader) 
    jaccard_idx = jaccard/len(loader)

    print('\nDice score: {}'.format(dice_s))
    print('Jaccard index: {}\n'.format(jaccard_idx))

    model.train()
    return dice_s, jaccard_idx
    
def save_preds_as_imgs(loader, model, device, folder = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Proyecto/saved_images'): 
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_grad(): # deshabilitar el cálculo y almacenamiento de gradientes en el grafo computacional de PyTorch
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(preds, f'{folder}/y_hat_{idx}.png') # almacenamiento de máscaras predichas
        y = torch.unsqueeze(y, 1).to(torch.float32)
        torchvision.utils.save_image(y, f'{folder}/y_{idx}.png') # almacenamiento de máscaras reales
        torchvision.utils.save_image(x, f'{folder}/x_{idx}.png') # almacenamiento de máscaras reales

        #masked_img= add_mask_to_rgb_image(x,preds)

        #torchvision.utils.save_image(masked_img, f'{folder}/masked_{idx}.png') # almacenamiento de máscaras reales

    
    model.train()

def add_mask_to_rgb_image(rgb_image_tensor, mask_tensor):
    # Apply the mask to the RGB image tensor
    masked_image_tensor = torch.where(mask_tensor > 0, torch.tensor([32,255, 0, 0]), rgb_image_tensor)

    return masked_image_tensor    #print(dice_score)
    #train_loss = [tensor.cpu() for tensor in train_loss]
    #train_dice = [tensor.cpu() for tensor in train_dice]


def save_metrics_to_csv(epoch, train_loss, train_dice, train_jaccard, dice_score, jaccard_score, filename):
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_dice': train_dice,
        'train_jaccard': train_jaccard,
        'dice_score': dice_score,
        'jaccard_score': jaccard_score
    }
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)


def save_preds_as_imgs2(loader, model, device, folder='/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Proyecto/saved_images'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Convert single-channel mask to RGB format
        y_rgb = y.repeat(1, 3, 1, 1)

        # Merge predicted mask with input image
        merged_img = x.clone()
        merged_img[:, :3] = torch.where(preds > 0, torch.tensor([1.0, 0.0, 0.0]), merged_img[:, :3])

        # Save the merged image
        vutils.save_image(merged_img, f'{folder}/merged_{idx}.png')

