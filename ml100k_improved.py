import os

import numpy as np
import pandas as pd
from torch.utils.data import random_split
import torch
from lit_data import BaseDataset

def read_data_ml100k_genre_added(data_dir="./csv_file") -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, 'user_item_genre_occupation.csv'))
    return data




class ML100KImproved(BaseDataset):
    def __init__(self, data_dir="./csv_file", normalize_rating=False) -> None:
        """MovieLens 100K for improved Matrix Factorization
        Each sample is a tuple of:
        - user_id: int
        - item_id: int
        - rating: float
        - occupation: string
        - 19 genre features: int, (0,1)
        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./csv_file"
        normalize_rating : bool, optional
            If True, rating is normalized to (0..1), by default False
        """
        super().__init__()
        self.normalize_rating = normalize_rating
        self.data_dir = data_dir
        self.df = read_data_ml100k_genre_added(data_dir)
        self.df.user_id -= 1
        self.df.item_id -= 1
        if normalize_rating:
            self.df.rating /= 5.0
        self.num_users = self.df.user_id.unique().shape[0]
        self.num_items = self.df.item_id.unique().shape[0]
        self.user_id = self.df.user_id.values
        self.item_id = self.df.item_id.values
        self.num_occupations = self.df.occupation.unique().shape[0] 
        self.num_genres = 19
        self.rating = self.df.rating.values.astype(np.float32)
        self.df.occupation = self.df.occupation.astype('category')
        categorical = self.df.occupation.cat.codes.to_numpy(dtype=np.int32)
        self.occupation = categorical
        self.genre = torch.from_numpy(self.df.iloc[:,-(len(self.df.columns)-5):].to_numpy(dtype=np.int32))


    def split(self, train_ratio=0.8):
        train_len = int(train_ratio*len(self))
        test_len = len(self) - train_len
        return random_split(self, [train_len, test_len])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.user_id[idx], self.occupation[idx], self.item_id[idx], self.genre[idx], self.rating[idx]