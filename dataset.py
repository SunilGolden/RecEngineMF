import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import datetime


class RatingsDataset(Dataset):
    def __init__(self, csv_file, split='train', user_set=None, item_set=None):
        self.data = pd.read_csv(csv_file)
        
        # Convert timestamp to unix timestamp
        self.data['timestamp'] = self.data['timestamp'].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
        
        # Split data into train, validation, and test sets based on timestamp
        self.data = self.data.sort_values(by='timestamp')
        n = len(self.data)
        train_size = int(0.7 * n)
        val_size = test_size = int(0.15 * n)

        if split == 'train':
            self.data = self.data[:train_size]
            self.user_set = set(self.data['userId'].unique())
            self.item_set = set(self.data['movieId'].unique())
        
        elif split == 'val':
            self.data = self.data[train_size:train_size+val_size]
            # Filter out user and item IDs that don't appear in the training set
            self.user_set = set(self.data['userId'].unique()) & user_set
            self.item_set = set(self.data['movieId'].unique()) & item_set
            self.data = self.data[self.data['userId'].isin(self.user_set) & self.data['movieId'].isin(self.item_set)]
        
        else:
            self.data = self.data[-test_size:]
            # Filter out user and item IDs that don't appear in the training set
            self.user_set = set(self.data['userId'].unique()) & user_set
            self.item_set = set(self.data['movieId'].unique()) & item_set
            self.data = self.data[self.data['userId'].isin(self.user_set) & self.data['movieId'].isin(self.item_set)]
            
        self.data = self.data.reset_index(drop=True)
        
        # Map user and item IDs to contiguous indices
        self.user_to_idx = {old_id: new_id for new_id, old_id in enumerate(self.data['userId'].unique())}
        self.item_to_idx = {old_id: new_id for new_id, old_id in enumerate(self.data['movieId'].unique())}
        
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        # Replace old IDs with new indices
        self.data['userId'] = self.data['userId'].apply(lambda x: self.user_to_idx[x])
        self.data['movieId'] = self.data['movieId'].apply(lambda x: self.item_to_idx[x])

        self.num_users = len(self.data['userId'].unique())
        self.num_items = len(self.data['movieId'].unique())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user = self.data.loc[idx, 'userId']
        item = self.data.loc[idx, 'movieId']
        rating = self.data.loc[idx, 'rating']
        timestamp = self.data.loc[idx, 'timestamp']
        
        sample = {'user': torch.tensor(user, dtype=torch.long),
                  'item': torch.tensor(item, dtype=torch.long),
                  'rating': torch.tensor(rating, dtype=torch.float32),
                  'timestamp': torch.tensor(timestamp, dtype=torch.long)}
        
        return sample
    
    def get_rated_items_by_user(self, user_id):
        """
        Get the items that a user has already rated.

        Args:
            user_id (int): User ID.

        Returns:
            list: List of item IDs that the user has rated.
        """
        rated_items = self.data[self.data['userId'] == user_id]['movieId'].tolist()
        return rated_items