# -*- coding: utf-8 -*-
"""
Desarrollado por: Wilson Javier Arenas López
Tema: UNET (CNNs & DCNNs)
Objetivo: Segmentar una imagen a través de la codificación (CNN) y decodificación (DCNN) de la misma, 
          teniendo como referencia la BD "Carvana Image Masking Challenge" y la arquitectura UNET 
          (https://arxiv.org/abs/1505.04597)
Parte: II (Construcción del modelo UNET)
Fecha: 24/04/2023
"""

import torch.nn as nn # para crear, definir y personalizar diferentes tipos de capas, modelos y criterios de pérdida en DL
import torch # para optimizar procesos de DL
import torchvision.transforms.functional as TF # para trasformaciones de tensores

class DoubleConv(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # Conv1 (flechas azules) (ver arquitectura UNET):
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True), 
            # Conv2 (flechas azules) (ver arquitectura UNET): 
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
            
    def forward(self, X): 
        return self.conv(X)
    
class UNET(nn.Module): 
    def __init__(self, in_channels = 3, out_channels = 1, feature_maps = [64, 128, 256, 512]): # asumiendo una salida binaria (máscara)
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # flechas rojas (ver arquitectura UNET)
        
        # CNN (downs): 
        for feat in feature_maps: 
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat # actualizamos con base en los feature_maps
            
        # Bottleneck:
        self.bottleneck = DoubleConv(feature_maps[-1], feature_maps[-1]*2)
        
        # DCNN (ups): 
        for feat in reversed(feature_maps):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size = 2, stride = 2)) # flechas verdes (ver arquitectura UNET)
            self.ups.append(DoubleConv(feat*2, feat)) # flechas azules (ver arquitectura UNET)
            
        # Output: 
        self.final_conv = nn.Conv2d(feature_maps[0], out_channels, kernel_size = 1) # flecha cian (ver arquitectura UNET)
        
    def forward(self, X):
        skip_connections = []
        
        # CNN (downs): 
        for down in self.downs: 
            X = down(X)
            skip_connections.append(X)
            X = self.pool(X)            
    
        # Bottleneck:
        X = self.bottleneck(X)
        skip_connections = skip_connections[::-1] # inversión del orden de las skip connections

        # DCNN (ups): 
        for idx in range(0, len(self.ups), 2): # el 2 es debido a la doble convolución
            X = self.ups[idx](X)
            skip_connection = skip_connections[idx//2] # módulo de la división
            
            if X.shape != skip_connection.shape: # para garantizar la igualdad de dimensiones
                X = TF.resize(X, size = skip_connection.shape[2:]) # solo me interesa altura y anchura
                
            concat_skip = torch.cat((skip_connection, X), dim = 1)
            X = self.ups[idx + 1](concat_skip)
            
        return self.final_conv(X)       
    
# def val_size(): 
#     X = torch.randn((3, 1, 255, 255))
#     model = UNET(in_channels = 1, out_channels = 1)
#     preds = model(X)
#     print('\nInput size: {}'.format(X.shape))
#     print('Output size: {}'.format(preds.shape))
    
# if __name__ == '__main__': 
#     val_size()