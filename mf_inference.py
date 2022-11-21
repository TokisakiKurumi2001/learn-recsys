from argparse import ArgumentParser
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import torch
from mf_improved import MatrixFactorization
from lit_data2 import  LitDataModule
from ml100k_improved import ML100KImproved

@dataclass
class MovieItem:
    name: str
    genres: List[str]

    def __str__(self):
        return f'Name: {self.name}. Genre: {", ".join(self.genres)}'

class MatrixFactorizationInference:
    def __init__(self, embedding_dims, num_users, num_items, num_occupations, num_genres, path: str=None):
        self.model = MatrixFactorization(embedding_dims, num_users, num_items, num_occupations, num_genres)
        if path is not None:
            self.model.load_state_dict(torch.load(path))

        self.df = pd.read_csv('csv_file/user_item_genre_occupation.csv')
        self.idx2occ = [
            'administrator',
            'artist',
            'doctor',
            'educator',
            'engineer',
            'entertainment',
            'executive',
            'healthcare',
            'homemaker',
            'lawyer',
            'librarian',
            'marketing',
            'none',
            'other',
            'programmer',
            'retired',
            'salesman',
            'scientist',
            'student',
            'technician',
            'writer',
        ]
        self.occ2idx = {occ: idx for idx, occ in enumerate(self.idx2occ)}
        df_genre = pd.read_csv('csv_file/u.genre', '|', names=['genre',  'index'])
        genre_lst = df_genre["genre"].tolist()
        df_item = pd.read_csv('csv_file/u.item', '|', names=["item_id", "name", "year", "dummy", "url", *genre_lst])
        df = df_item.drop(["year", "dummy", "url"], axis=1)
        self.df_item = df

    def search_for_occupation(self, user_id: int) -> int:
        return self.occ2idx[self.df[self.df['user_id']==user_id]['occupation'].unique().item()]

    def find_genre(self, item_id: int) -> int:
        return torch.from_numpy(self.df[self.df['item_id'] == item_id].iloc[0, -(len(self.df.columns)-5):].to_numpy(dtype=np.int32))

    def within_range(self, value: float, threshold: int) -> bool:
        return abs(threshold - value) < 0.4

    def id2movie(self, item_id: int) -> MovieItem:
        row = self.df_item[self.df_item['item_id'] == item_id]
        movie_name = row.name.item()
        genres = row.columns[3:] # get rid of unknown
        movie_genres = []
        for genre in genres:
            if row[genre].item() == 1:
                movie_genres.append(genre)
        return MovieItem(movie_name, movie_genres)

    def inference(self, user_id: int, max_num_items: int):
        self.model.eval()
        user_id -= 1 # id for user_id starts from 0
        occ_id = self.search_for_occupation(user_id + 1) # reading from CSV id starts from 1 --> align

        user_id = torch.tensor([user_id])
        occ_id = torch.tensor([occ_id])
        rcd_ids = []
        
        for item_id in range(max_num_items):
            genre_ids = self.find_genre(item_id + 1).unsqueeze(0)
            item_id_tensor = torch.tensor([item_id])
            rating = self.model(user_id, item_id_tensor, occ_id, genre_ids).item()
            if self.within_range(rating, 3):
                rcd_ids.append(item_id + 1)
        
        movies = []
        for item_id in rcd_ids:
            if item_id == 267:
                continue
            movies.append(self.id2movie(item_id))
        return movies

def main(args):
    data = LitDataModule(ML100KImproved("./csv_file"), batch_size=args.batch_size, num_workers=24)
    data.setup()
    inference = MatrixFactorizationInference(embedding_dims=args.embedding_dims,  
        num_users=data.num_users, num_items=data.num_items, num_occupations=data.num_occupations, num_genres = data.num_genres,
        path=args.path)
    user_id = args.user_id
    movies = inference.inference(user_id, data.num_items)
    with open(f'user_{user_id}_rcd.txt', 'w+') as file:
        for movie in movies:
            file.write(f'{movie}\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--user_id", type=int, default=1)
    parser.add_argument("--path", type=str, default="models/our.pt")
    args = parser.parse_args()
    main(args)
        