from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from lit_data2 import  LitDataModule
from lit_model import LitModel
from ml100k_improved import ML100KImproved

class EmbeddingMulti(nn.Module):
    def __init__(self, num_genres, embedding_dims, sparse=False):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.embedding = nn.Embedding(num_genres, embedding_dims, sparse=sparse)
        

    def forward(self, input):
        """ input: tensor of shape (batch_size, num_genres)"""
        device = input.get_device()
        result = torch.empty(size=(input.shape[0], self.embedding_dims), device=device)
        for i in range(input.shape[0]): # batch_size
            indices = torch.nonzero(input[i], as_tuple=True)[0]
            embeddings = [self.embedding(x) for _, x in enumerate(indices)]
            vec = torch.mean(torch.stack(embeddings))
            result[i,:] = vec

        return result


class MatrixFactorization(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items, num_occupations, num_genres,
                 sparse=False, **kwargs):
        super().__init__()
        self.sparse = sparse
        self.embedding_dims = embedding_dims
        self.user_embedding = nn.Embedding(num_users, embedding_dims, sparse=sparse) # matrix P
        self.user_bias = nn.Embedding(num_users, 1, sparse=sparse)                   # bias P
        
        self.item_embedding = nn.Embedding(num_items, embedding_dims, sparse=sparse) # matrix Q
        self.item_bias = nn.Embedding(num_items, 1, sparse=sparse)                   # bias P
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dims, sparse=sparse) 
        self.occupation_bias = nn.Embedding(num_occupations, 1, sparse=sparse)
        

        # create many genre embeddings for item
        # Approach: sum(many embeddings) for one item
        # Approach: mean //
        # Approach:  unknown and other as embeddings
        
        self.genre_embedding = EmbeddingMulti(num_genres, embedding_dims, sparse=sparse)
        self.genre_bias = EmbeddingMulti(num_genres, 1, sparse=sparse)

        for param in self.parameters():
            nn.init.normal_(param, std=0.01)

    def forward(self, user_id, item_id, occupation, genre):
        Q = self.user_embedding(user_id)
        bq = self.user_bias(user_id).flatten()

        I = self.item_embedding(item_id)
        bi = self.item_bias(item_id).flatten()

        O = self.occupation_embedding(occupation)
        bo = self.occupation_bias(occupation).flatten()
        
        G = self.genre_embedding(genre)
        bg = self.genre_bias(genre).flatten()
        return (Q*I).sum(-1) + (O*Q).sum(-1) + (I*G).sum(-1) + bq + bi + bo + bg




class LitMF(LitModel):
    def get_loss(self, pred_ratings, batch):
        return F.mse_loss(pred_ratings, batch[-1])

    def update_metric(self, m_outputs, batch, partition):
        gt = batch[-1]
        if partition == 'train':
            self.train_rmse.update(m_outputs, gt)
        else:
            self.valid_rmse.update(m_outputs, gt)

    def forward(self, batch):
        user_ids, user_occupation, item_ids, item_genre = batch[:4]
        return self.model(user_ids, item_ids, user_occupation, item_genre)
        


def main(args):

    data = LitDataModule(ML100KImproved("./csv_file"), batch_size=args.batch_size, num_workers=24)
    data.setup()
    model = LitMF(MatrixFactorization, embedding_dims=args.embedding_dims,  
        num_users=data.num_users, num_items=data.num_items, num_occupations=data.num_occupations, num_genres = data.num_genres, sparse=False
        )
    
    # wandb_logger = WandbLogger(project="proj_dummy")
    wandb_logger = WandbLogger(project="proj_recsys")
    trainer = pl.Trainer.from_argparse_args(
        args,
        devices=2,
        accelerator="gpu",
        strategy="ddp",
        max_epochs=30,
        logger=wandb_logger)

    train_dataloader = data.train_dataloader()
    valid_dataloader = data.val_dataloader()
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
