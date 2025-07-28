import os
import torch
from training import train_val
from model import getmodel
from config import image_dir, mask_dir
from data_loader import get_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 20
    val_step = 1
    batch_size = 128
    
    dataset_path = os.path.join(image_dir, mask_dir)
    train_loader, val_loader = get_dataloaders(dataset_path, batch_size)
    
    model = getmodel().to(device)
    #model.load_state_dict(torch.load("stage1_11_model_weights_v2.pth"))
    print("Model loaded successfully.")
    
    save_path = 'weights/'
    os.makedirs(save_path, exist_ok=True)
    
    train_val(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        val_step,
        save_path
    )
        
if __name__ == "__main__":
    main()