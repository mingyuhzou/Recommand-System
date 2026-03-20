import torch
from pure_eval.my_getattr_static import user_method_descriptor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import model_cfg,data_cfg
import pickle
import numpy as np

class TwoTowers(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_units,dropout=model_cfg['dropout']):
        super().__init__()

        self.dropout=dropout
        self.embed_dim = embed_dim
        self.mlp_hidden_units = mlp_hidden_units

        embedding_matrix = np.load(data_cfg["word2emb"])

        with open(data_cfg["mid2idx"], "rb") as f:
            mid2idx = pickle.load(f)
        with open(data_cfg["uid2idx"], "rb") as f:
            uid2idx = pickle.load(f)
        with open(data_cfg["genre2idx"], "rb") as f:
            genre2idx = pickle.load(f)
        with open(data_cfg["zip2idx"], "rb") as f:
            zipcode2idx = pickle.load(f)



        self.movie_embedding = nn.Embedding(
            num_embeddings=len(mid2idx) + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.genre_embedding = nn.Embedding(
            num_embeddings=len(genre2idx) + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.title_embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )


        self.user_embedding = nn.Embedding(
            num_embeddings=len(uid2idx) + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.age_embedding = nn.Embedding(
            num_embeddings=model_cfg["age_num"] + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.occupation_embedding = nn.Embedding(
            num_embeddings=model_cfg["occupation_num"] + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.gender_embedding = nn.Embedding(
            num_embeddings=2 + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.zipcode_embedding = nn.Embedding(
            num_embeddings=len(zipcode2idx) + 1,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.user_tower=self.mlp_layers(5*embed_dim,mlp_hidden_units,embed_dim)
        self.movie_tower=self.mlp_layers(3*embed_dim,mlp_hidden_units,embed_dim)

        self._init_weights()

    def mlp_layers(self,input_dim,hidden_layers,output_dim):
        layers=[]
        prev_dim=input_dim
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(prev_dim,hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim=hidden_units
        layers.append(nn.Linear(prev_dim,output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)

    def forward(self, input):
        user_output = self.get_user_embedding(input)
        movie_output = self.get_item_embedding(input)
        return user_output, movie_output

    def get_user_embedding(self, input):
        user = self.user_embedding(input["uid"])
        age = self.age_embedding(input["age"])
        occupation = self.occupation_embedding(input["occupation"])
        gender = self.gender_embedding(input["gender"])
        zipcode = self.zipcode_embedding(input["zipcode"])

        user_embedding = torch.cat([user, age, occupation, gender, zipcode], dim=1)
        user_output = self.user_tower(user_embedding)

        return user_output

    def get_item_embedding(self, input):
        movie = self.movie_embedding(input["mid"])

        genre = self.genre_embedding(input["genres"])
        genre = genre.mean(dim=1)

        title = self.title_embedding(input["title"])
        title = title.mean(dim=1)

        movie_embedding = torch.cat([movie, genre, title], dim=1)
        movie_output = self.movie_tower(movie_embedding)

        return movie_output
