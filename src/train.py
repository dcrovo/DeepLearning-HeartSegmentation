# -*- coding: utf-8 -*-
"""

"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch # para optimizar procesos de DL
import multiprocessing # para conocer el número de workers (núcleos)
import albumentations as A # para emplear data augmentation
from albumentations.pytorch import ToTensorV2 # para convertir imágenes y etiquetas en tensores de Pytorch
import torch.optim as optim # para optimización de parámetros
import torch.nn as nn # para crear, definir y personalizar diferentes tipos de capas, modelos y criterios de pérdida en DL
from tqdm import tqdm # para agregar barras de progreso a bucles o iteraciones de larga duración
from utils import ( # .py previamente desarrollado
                       get_loaders,
                       load_checkpoint,
                       save_checkpoint,
                       perf,
                       save_preds_as_imgs,
                       save_metrics_to_csv,
                       compute_jaccard_index,
                      )
#from loss_fn import dice_coefficient_loss, dice_coefficient
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
from Unet import UNET
# Hiperparámetros preliminares: 
LEARNING_RATE = 0.5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 
NUM_EPOCHS = 50
NUM_WORKERS = multiprocessing.cpu_count()
IMAGE_HEIGHT = 192 # 1280 px.
IMAGE_WIDTH = 192 # 1918 px.
PIN_MEMORY = True # almacena en la memoria fija del sistema una copia de los datos cargados en la memoria temporal de Python
LOAD_MODEL = False
CHECKPOINT_PATH = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/checkpoints/MANET_my_checkpoint.pth.tar'
TRAIN_IMG_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/images'
TRAIN_MAKS_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/masks'
VAL_IMG_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/val_images'
VAL_MASK_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/val_masks'
PREDS_PATH = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/saved_images'
METRICS_PATH = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/scores/MANET_.csv'

def train_func(train_loader, model, optimizer, loss_fn, scaler): 
    
    p_bar = tqdm(train_loader) # progress bar
    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    dice = 0
    jaccard = 0
    for batch_idx, (data, targets) in enumerate(p_bar):
        
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE) # agregar una nueva dimensión en la posición 1 del tensor "targets"
        #if not torch.all(targets==0):
        # Forward pass: 
        with torch.cuda.amp.autocast(): #para realizar operaciones de cálculo de punto flotante de precisión mixta (16 y 32 bits) en modelos de DL
            preds = model(data)
            loss = loss_fn(preds, targets)
            with torch.no_grad():
                p = torch.sigmoid(preds)
                p = (p > 0.5).float()
                sum = targets.sum() 
                if((p.sum() == 0) and (targets.sum() == 0)):
                    jaccard = 1
                    dice = (2*(p*targets).sum())/((p + targets).sum() + 1e-10) + 1
                else:
                    jaccard = compute_jaccard_index(p, targets)
                    dice = (2*(p*targets).sum())/((p + targets).sum() + 1e-10)
            
        
        # Backward pass: 
        optimizer.zero_grad() 
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update() 
        running_loss += loss.item()
        running_dice += dice
        running_jaccard += jaccard
        
        # Actualización progress bar: 
        p_bar.set_postfix(loss = loss.item(), dice=dice.item(), jaccard=jaccard)
        torch.cuda.empty_cache()

    epoch_loss = running_loss/len(train_loader)
    epoch_dice = running_dice/len(train_loader)
    epoch_jaccard = running_jaccard/len(train_loader)
    return epoch_loss, epoch_dice, epoch_jaccard
        
def main():
    x_start = 50
    x_end = 462
    y_start = 50
    y_end = 462 
    train_transforms = A.Compose(
                                 [
                                  A.Crop(x_start, y_start, x_end, y_end, always_apply= True),
                                  A.CLAHE(p=0.1),
                                  A.GaussNoise(p=0.1),
                                  A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH), 
                                  A.Rotate(limit = 30, p = 0.3), 
                                  A.HorizontalFlip(p = 0.3), 
                                  A.VerticalFlip(p = 0.3), 
                                  A.Normalize(
                                              mean = [0.0, 0.0, 0.0],
                                              std = [1.0, 1.0, 1.0], 
                                              max_pixel_value = 255.0
                                             ),
                                  ToTensorV2(), # conversión a tensor de Pytorch
                                 ],
                                )
    
    val_transforms = A.Compose(
                               [A.Crop(x_start, y_start, x_end, y_end, always_apply= False),
                                A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH), 
                                A.Normalize(
                                            mean = [0.0, 0.0, 0.0],
                                            std = [1.0, 1.0, 1.0], 
                                            max_pixel_value = 255.0
                                         ),
                              ToTensorV2(), # conversión a tensor de Pytorch
                             ],
                            )
    
    train_loader, val_loader = get_loaders( # cargar y preprocesar los datos de entrenamiento y validación con base en los formatos tensoriales
                                           TRAIN_IMG_DIR, 
                                           TRAIN_MAKS_DIR, 
                                           VAL_IMG_DIR, 
                                           VAL_MASK_DIR,
                                           BATCH_SIZE,
                                           train_transforms,
                                           val_transforms, 
                                           NUM_WORKERS,
                                           w_level=40, 
                                           w_width=350,
                                           pin_memory= PIN_MEMORY, 
                                           normalize=False, 
                                           
                                          )

    #model = UNET(3,1).to(DEVICE)
    #preprocess_input = get_preprocessing_fn('mit_b3', pretrained='imagenet')
    model = smp.MAnet(   
                encoder_name="resnext101_32x4d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="ssl",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,
                decoder_use_batchnorm=True,
                ).to(DEVICE)
    
    if LOAD_MODEL == True: # existe un modelo entrenado preliminarmente
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)
                                      
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn =  smp.losses.DiceLoss(smp.losses.BINARY_MODE, log_loss=True, eps=1e-10, from_logits=True) # Binary Cross Entropy with logits loss   
    scaler = torch.cuda.amp.GradScaler() # ajustar de manera dinámica la escala de los gradientes durante el entrenamiento para reducir el tiempo de entrenamiento mientras se mantiene la precisión de los resultados                        
    train_loss = []
    train_dice = []
    train_jaccard = []
    dice_score =[]
    jaccard_score = []
    # Entrenamiento: 
    for epoch in tqdm(range(NUM_EPOCHS)): 
        epoch_loss, dice_train, jaccard_train = train_func(train_loader, model, optimizer, loss_fn, scaler)
        train_loss.append(epoch_loss)
        train_dice.append(dice_train.detach().cpu().numpy())
        train_jaccard.append(jaccard_train)
        
        # Creación de checkpoint: 
        checkpoint = {
                      'state_dict': model.state_dict(), # estado de los parámetros 
                      'optimizer': optimizer.state_dict(), # estado de los gradientes            
                     }
        save_checkpoint(checkpoint, CHECKPOINT_PATH)
        
        # Rendimiento: 
        dice, jaccard = perf(val_loader, model, DEVICE)
        dice_score.append(dice.detach().cpu().numpy())
        jaccard_score.append(jaccard)
        
        # Almacenamiento de predicciones: 
        save_preds_as_imgs(val_loader, model, DEVICE,PREDS_PATH)
        save_metrics_to_csv(epoch,epoch_loss, dice_train.detach().cpu().numpy(), jaccard_train, dice.detach().cpu().numpy(), jaccard, METRICS_PATH)



    #train_jaccard = [tensor.cpu()  for tensor in train_jaccard]
    dice_score = [tensor.cpu() for tensor in dice_score]
    #jaccard_score = [tensor.cpu() for tensor in jaccard_score]

    fig, axes = plt.subplots(1,3)
    fig.set_figheight(6)
    fig.set_figwidth(18)
    axes[0].plot(train_loss)
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    axes[1].plot(train_dice)
    axes[1].set_title('Dice Score in train set')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Dice Score')
    
    axes[2].plot(train_jaccard)
    axes[2].set_title('Jaccard Index in train set')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Jaccard Index')
    plt.savefig('/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/plots/Training_MANET.png')
    #plt.show()


    fig2, axes2 = plt.subplots(1,2)
    fig2.set_figheight(6)
    fig2.set_figwidth(18)

    axes2[0].plot(dice_score)
    axes2[0].set_title('Dice Score in validation set')
    axes2[0].set_xlabel('Epochs')
    axes2[0].set_ylabel('Dice Score')
    
    axes2[1].plot(jaccard_score)
    axes2[1].set_title('Jaccard Index in validation set')
    axes2[1].set_xlabel('Epochs')
    axes2[1].set_ylabel('Jaccard Index')
    plt.savefig('/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/plots/Validation_MANET.png')
    #plt.show()


if __name__ == '__main__': 
    main()