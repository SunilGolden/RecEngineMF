import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def reset_random(random_seed=42):
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(random_seed)

	
def get_device():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	return device


def calc_loss(model, loader):
    device = get_device()

    model.to(device)
    model.eval()
    
    loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            ratings = batch['rating'].to(device)
            
            y_hat = model(users, items)
            batch_loss = F.mse_loss(y_hat, ratings)
            loss += batch_loss.item()
            num_batches += 1
    
    return loss / num_batches


def train_epochs(model, 
                 train_loader, 
                 val_loader, 
                 epochs=10, 
                 lr=0.01, 
                 weight_decay=0.0, 
                 step_size=10, 
                 gamma=0.1, 
                 patience=3, 
                 model_name='mf_model.pth',
                 metrics_csv_name='metrics.csv',
                 verbose=True):
    device = get_device()

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    best_val_loss = float('inf')
    best_model = None
    early_stopping_counter = 0
    
    csv_logger = csv.writer(open(metrics_csv_name, "w"))
    csv_logger.writerow(["Epoch", "Train Loss", "Val Loss"])
    
    for i in range(epochs):
        model.train()
        
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            ratings = batch['rating'].to(device)
            
            y_hat = model(users, items)
            batch_loss = F.mse_loss(y_hat, ratings)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
            num_batches += 1
            
            del(users)
            del(items)
            del(ratings)
            del(y_hat)
        
        val_loss = calc_loss(model, val_loader)
        csv_logger.writerow([i+1, train_loss / num_batches, val_loss])
        
        if verbose:
        	print('Epoch: %d\tTrain Loss: %.4f\t Val Loss: %.4f'% (i+1, train_loss / num_batches, val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0
            torch.save(model, model_name)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print('Early stopping at epoch %d. Best val loss: %.4f'% (i+1, best_val_loss))
                break


def test(model_path, test_loader):
    mae, rmse = 0.0, 0.0

    device = get_device()

    model = torch.load(model_path)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            ratings = batch['rating'].to(device)
            
            predictions = model(users, items)
            
            mae += calculate_mae(predictions.cpu().numpy(), ratings.cpu().numpy())
            rmse += calculate_rmse(predictions.cpu().numpy(), ratings.cpu().numpy())

            del(users)
            del(items)
            del(ratings)
            del(predictions)

    print('MAE:', mae)
    print('RMSE:', rmse)


def get_top_k_recommendations(model, user_to_idx, idx_to_item, rated, user_id, k=10):
    if user_id not in user_to_idx:
        print('This user was not present in the Training set')
        return -1

    device = get_device()

    # user_id to user_index
    user_idx = user_to_idx[user_id]
    
    u = torch.tensor(user_idx).unsqueeze(dim=0).to(device)
    v = torch.arange(model.item_emb.weight.size(0)).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(u, v).cpu().numpy()
        
        max_indices = np.argsort(predictions)
        top_k_item_idx = np.array([x for x in max_indices if x not in np.array(rated)])[:k]

        # item_id to item_index
        top_k_items = [idx_to_item[i] for i in top_k_item_idx]
        
        return top_k_items


def calculate_mae(predictions, actual_ratings):
    """Calculate the mean absolute error (MAE) between predicted and actual ratings"""
    return np.mean(np.abs(predictions - actual_ratings))


def calculate_rmse(predictions, actual_ratings):
    """Calculate the root mean squared error (RMSE) between predicted and actual ratings"""
    return np.sqrt(np.mean((predictions - actual_ratings) ** 2))


def plot_loss_curve_from_csv(metrics_csv_path, patience=None, save=True, file_name='loss_curve.png'):
    df = pd.read_csv(metrics_csv_path)

    epoch = df['Epoch']
    train_loss = df['Train Loss']
    val_loss = df['Val Loss']

    loss_df = pd.DataFrame({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss})

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['Epoch'], loss_df['Train Loss'], label='Train Loss')
    plt.plot(loss_df['Epoch'], loss_df['Val Loss'], label='Val Loss')
    
    if patience:
        plt.plot([len(epoch)-patience, len(epoch)-patience], plt.ylim(), label='Early Stopped', linestyle='--', color='black')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if save:
        plt.savefig(file_name)

    plt.show()