import os
import torch
from tqdm import tqdm

def train_val(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    val_step,
    save_path
):
    save_path = 'weights/'
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        
        train_correct = 0
        train_total = 0
        train_loss_total = 0
        
        tq = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (data, target, _) in enumerate(tq):
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            
            loss = criterion(output, target)
            train_loss_total += loss.item()
            
            preds = output.argmax(dim=1)
            valid = target
            
            train_correct += (preds[valid] == target[valid])/sum().item()
            train_total += valid.sum().item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            tq.set_postfix(loss=loss.item(), acc=train_acc)
            
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss_total / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        if epoch%val_step == 0:
            model.eval()
            
            val_loss_total = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for val_data, val_target, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                    val_data = val_data.to(device)
                    val_target = val_target.to(device)
                    
                    val_output = model(val_data)[0]
                    
                    val_loss = criterion(val_output, val_target)
                    val_loss_total += val_loss.item()
                    
                    preds = val_output.argmax(dim=1)
                    valid = val_target
                    val_correct += (preds[valid] == val_target[valid])
                    val_total += valid.sum().item()
                
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss_total / len(val_loader)
            print(f"[Val] Epoch {epoch+1} - Validation Loss = {avg_val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")
            
        torch.save(model.state_dict(), os.path.join(save_path, f"{epoch+1}_model_weights.pth"))
            
            
            
    