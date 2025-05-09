import keras
import tensorflow as tf
import torchvision
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn,  maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

def get_maskrcnn(num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=num_classes)
    dtype = torch.float32
    model.to(device=device, dtype=dtype);

    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'
    return model

def res_block(x, filters, kernel_size: tuple = (3, 3), strides = 1, activation = 'relu'):
    shortcut = x
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(activation)(x)

    return x


def build_unet(input_shape: tuple) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    num_filters = 64

    skip_connections = []
    x = inputs

    # Encoder
    for _ in range(4):
        for _ in range(3):
            x = keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same')(x)
            x = keras.layers.Activation('relu')(x)
        skip_connections.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        num_filters *= 2

    # Bottleneck
    for _ in range(3):
        x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding="same")(x)
        x = keras.layers.Activation('relu')(x)
    
    # Decoder
    for i in range(4):
        x = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=2, strides=2, padding="same")(x)
        num_filters //= 2
        x = keras.layers.Concatenate()([x, skip_connections[-(i + 1)]])
        for _ in range(3):
            x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding="same")(x)
            x = keras.layers.Activation('relu')(x)
        
    outputs = keras.layers.Conv2DTranspose(filters=2, kernel_size=(2, 2))(x)

    return keras.Model(inputs, outputs)
    

build_unet((224, 224, 3)).summary()
        
