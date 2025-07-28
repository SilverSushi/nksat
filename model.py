import satlaspretrain_models

def getmodel():
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Sentinel2_Resnet50_SI_RGB", 
                                                 fpn=True, 
                                                 head=satlaspretrain_models.Head.SEGMENT,
                                                 num_categories=11, 
                                                 device='cuda')
    return model