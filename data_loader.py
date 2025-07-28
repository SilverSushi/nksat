from torch.utils.data import dataloader
from torchvision import transforms
import torchvision.transforms as T
from dataset import SegmentationDataset

def get_dataloaders(dataset_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mask_transform = T.Compose([
        T.PILToTensor(),
        lambda x: x.squeeze(0).long()
    ])
    
    train_dataset = SegmentationDataset(dataset_path, 
                                        val=False, 
                                        transform=transform, 
                                        target_transform=mask_transform
                                        )
    
    val_dataset = SegmentationDataset(dataset_path, 
                                      val=True, 
                                      transform=transform, 
                                      target_transform=mask_transform
                                      )
    
    train_loader = dataloader.DataLoader(train_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         num_workers=4
                                         )
    
    val_loader = dataloader.DataLoader(val_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       num_workers=4
                                       )
    
    return train_loader, val_loader