import torch
from pure_eval.my_getattr_static import user_method_descriptor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import model_cfg,data_cfg
import pickle
import numpy as np

class TwoTowers(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_units, dropout=model_cfg["dropout"]):
        super().__init__()

        self.dropout = dropout
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
            torch.tensor(embedding_matrix, dtype=torch.float32),
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

        # 用户侧 5 个field，物品侧 3 个field
        self.user_field_num = 5
        self.item_field_num = 3

        # SENet reduction ratio
        reduction = 2

        # user SENet
        self.user_senet = nn.Sequential(
            nn.Linear(self.user_field_num, max(1, self.user_field_num // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, self.user_field_num // reduction), self.user_field_num),
            nn.Sigmoid()
        )

        # item SENet
        self.item_senet = nn.Sequential(
            nn.Linear(self.item_field_num, max(1, self.item_field_num // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, self.item_field_num // reduction), self.item_field_num),
            nn.Sigmoid()
        )

        self.user_tower = self.mlp_layers(self.user_field_num * embed_dim, mlp_hidden_units, embed_dim)
        self.movie_tower = self.mlp_layers(self.item_field_num * embed_dim, mlp_hidden_units, embed_dim)

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

        user_fields=torch.stack([user,age,occupation,gender,zipcode],dim=1) # [B, 5, D]
        user_fields=self._apply_senet(user_fields,self.user_senet)

        # user_embedding = torch.cat([user, age, occupation, gender, zipcode], dim=1)
        user_embeddings=user_fields.view(user_fields.size(0),-1) # [B,5*D]
        user_output = self.user_tower(user_embeddings)

        return user_output

    def get_item_embedding(self, input):
        movie = self.movie_embedding(input["mid"])

        genre = self.genre_embedding(input["genres"])
        genre = genre.mean(dim=1)

        title = self.title_embedding(input["title"])
        title = title.mean(dim=1)

        item_fields=torch.stack([movie,genre, title], dim=1)

        item_fields=self._apply_senet(item_fields,self.item_senet) # [B, 3, D]

        # movie_embedding = torch.cat([movie, genre, title], dim=1)
        movie_embedding=item_fields.view(item_fields.size(0),-1)
        movie_output = self.movie_tower(movie_embedding)

        return movie_output

    def _apply_senet(self,embeddings,senet_layer):
        z=embeddings.mean(dim=-1) # # [B, F, D] -> [B, F]

        a=senet_layer(z)
        a=a.unsqueeze(-1)

        v=embeddings*a
        return v

    @torch.no_grad()
    def debug_senet_weights(self, input):
        """返回senet权重"""
        self.eval()

        # ===== user =====
        user = self.user_embedding(input["uid"])
        age = self.age_embedding(input["age"])
        occupation = self.occupation_embedding(input["occupation"])
        gender = self.gender_embedding(input["gender"])
        zipcode = self.zipcode_embedding(input["zipcode"])

        user_fields = torch.stack([user, age, occupation, gender, zipcode], dim=1)
        z_user = user_fields.mean(dim=-1)  # [B, 5]
        a_user = self.user_senet(z_user)  # [B, 5]

        # ===== item =====
        movie = self.movie_embedding(input["mid"])

        genre = self.genre_embedding(input["genres"]).mean(dim=1)
        title = self.title_embedding(input["title"]).mean(dim=1)

        item_fields = torch.stack([movie, genre, title], dim=1)
        z_item = item_fields.mean(dim=-1)  # [B, 3]
        a_item = self.item_senet(z_item)  # [B, 3]

        return a_user, a_item



