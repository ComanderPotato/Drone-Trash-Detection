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

## From that github repo
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
