import torch
import torch.nn as nn

class sdf_transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, ff_dim):
        super(sdf_transformer, self).__init__()

        self.linear_embedding = nn.Linear(3*input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.feedforward = nn.Sequential(
                            nn.Linear(embed_dim, ff_dim),
                            nn.ReLU(),
                            nn.Linear(ff_dim, output_dim))

    def forward(self, input_tensor):
        input_tensor = torch.cat((input_tensor, torch.sin(input_tensor), torch.cos(input_tensor)), dim=-1)

        x = self.linear_embedding(input_tensor)
        #x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        #x = x.permute(1, 0, 2)[:, -1]
        mindist = self.feedforward(x)
        return mindist
