import os
import torch
from torch.utils.data import dataloader
from torchvision import transforms
import torchvision.transforms as T
from dataset import SegmentationDataset
from training import train, validate
from model import getmodel
from config import image_dir, mask_dir

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 128
    
    dataset_path = os.path.join(image_dir, mask_dir)
    
    train_loader, val_loader = get_dataloaders(dataset_path, batch_size)
    
    model = getmodel(num_classes=len(os.listdir(mask_dir))).to(device)
    print("Model loaded successfully.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    num_epochs = 20
    
    save_path = 'weights_aug/'
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        torch.save(model.state_dict(), os.path.join(save_path, f"{epoch+1}_model_weights.pth"))
        
if __name__ == "__main__":
    main()