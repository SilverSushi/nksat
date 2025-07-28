import torch
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, _ in loop:
        images, masks = images.to(device), masks.to(device)
        
        optimzer = optimizer.zero_grad()
        outputs = model(images)[0]
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimzer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / len(dataloader))
        
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for images, masks, _ in loop:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)[0]
            
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    
    return running_loss / len(dataloader)