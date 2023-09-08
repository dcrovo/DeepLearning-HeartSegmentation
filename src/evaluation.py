from utils import ( # .py previamente desarrollado
                       get_loaders,
                       load_checkpoint,
                       save_checkpoint,
                       perf,
                       save_preds_as_imgs,
                       save_metrics_to_csv,
                       compute_jaccard_index,
                      )
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import multiprocessing # para conocer el número de workers (núcleos)
import albumentations as A # para emplear data augmentation
from albumentations.pytorch import ToTensorV2 # para convertir imágenes y etiquetas en tensores de Pytorch
import torch.optim as optim # para optimización de parámetros
import segmentation_models_pytorch as smp


CHECKPOINT_PATH = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/checkpoints/MANET_my_checkpoint.pth.tar'
TEST_IMAGES = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/test_images'
TEST_MASK = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/test_masks'
TRAIN_IMG_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/images'
TRAIN_MAKS_DIR = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/2d_data/masks'
IMAGE_HEIGHT = 192 # 1280 px.
IMAGE_WIDTH = 192 # 1918 px.
PREDS_PATH = '/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Heart_Segmentation/saved_images_test'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_training_set(model, loader, device):
    model.eval()

    dice_score = 0.0
    jaccard = 0.0
    num_samples = 0

    with torch.no_grad():
        for data, y in loader:
            data = data.to(device)
            y = y.float().unsqueeze(1).to(device)
            preds = model(data)
            preds = torch.sigmoid(model(data))
            
            preds = (preds > 0.5).float()
            
            if((preds.sum() == 0) and (y.sum() == 0)):
                 jaccard +=1
                 dice_score +=1
            else:
                jaccard += compute_jaccard_index(preds, y)
                dice_score += (2*(preds*y).sum())/((preds + y).sum() + 1e-10)
            

    dice_s = dice_score/len(loader) 
    jaccard_idx = jaccard/len(loader)


    return dice_s, jaccard_idx

def main():
    x_start = 50
    x_end = 462
    y_start = 50
    y_end = 462 
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
    train_loader, test_loader = get_loaders( # cargar y preprocesar los datos de entrenamiento y validación con base en los formatos tensoriales
                                           TRAIN_IMG_DIR, 
                                           TRAIN_MAKS_DIR, 
                                           TEST_IMAGES, 
                                           TEST_MASK,
                                           16,
                                           None,
                                           val_transforms, 
                                           16,
                                           w_level=40, 
                                           w_width=350,
                                           pin_memory= True, 
                                           normalize=False, 
                                           
                                          )
    model = smp.MAnet(   
                encoder_name="mit_b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,
                decoder_use_batchnorm=True,
                ).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_PATH), model)


    dice_score, jaccard_index = evaluate_training_set(model, test_loader, DEVICE)
    print(f"Average Dice Score (Training Set): {dice_score:.4f}")
    print(f"Average Jaccard Index (Training Set): {jaccard_index:.4f}")
    save_preds_as_imgs(test_loader, model, DEVICE,PREDS_PATH)

if __name__ == '__main__': 
    main()