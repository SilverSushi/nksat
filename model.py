import satlaspretrain_models
import os
from config import mask_dir

def getmodel():
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Sentinel2_Resnet50_SI_RGB", 
                                                 fpn=True, 
                                                 head=satlaspretrain_models.Head.SEGMENT,
                                                 num_categories=len(os.listdir(mask_dir)), 
                                                 device='cuda')
    return model