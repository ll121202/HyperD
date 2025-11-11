import torch
import torch.nn as nn
import torch.nn.functional as F


class stfe(nn.Module):
    def __init__(self, num_nodes, seq_len, pred_len, embed_size, hidden_size, fc_hidden_size):
        super(stfe, self).__init__()
        self.scale = 0.02
        self.feature_size = num_nodes
        self.seq_length = seq_len
        self.sparsity_threshold = 0.01

        self.embeddings = nn.Parameter(torch.randn(1, embed_size))

        self.spatial_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.spatial_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        self.temporal_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.temporal_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        self.fc = nn.Sequential(
            nn.Linear(seq_len * embed_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, pred_len)
        )


    def tokenEmb(self, x):
        # x: (B, T, N)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def C_MLP_s(self, x):
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.C_MLP(x, self.spatial_r1, self.spatial_i1, self.spatial_r2, self.spatial_i2, self.spatial_rb1,
                        self.spatial_rb2, self.spatial_ib1, self.spatial_ib2)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        return x

    def C_MLP_t(self, x):
        x = x.transpose(1, 2)
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.C_MLP(x, self.temporal_r1, self.temporal_i1, self.temporal_r2, self.temporal_i2, self.temporal_rb1,
                        self.temporal_rb2, self.temporal_ib1, self.temporal_ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        x = x.transpose(1, 2)
        return x

    def C_MLP(self, x, r1, i1, r2, i2, rb1, rb2, ib1, ib2):
        o1_real = F.relu(
            torch.einsum('bijd,df->bijf', x.real, r1) - \
            torch.einsum('bijd,df->bijf', x.imag, i1) + \
            rb1
        )
        o1_imag = F.relu(
            torch.einsum('bijd,df->bijf', x.imag, r1) + \
            torch.einsum('bijd,df->bijf', x.real, i1) + \
            ib1
        )

        o2_real = F.relu(
            torch.einsum('bijf,fd->bijd', o1_real, r2) - \
            torch.einsum('bijf,fd->bijd', o1_imag, i2) + \
            rb2
        )
        o2_imag = F.relu(
            torch.einsum('bijf,fd->bijd', o1_imag, r2) + \
            torch.einsum('bijf,fd->bijd', o1_real, i2) + \
            ib2
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)


    def forward(self, x):
        B, T, N = x.shape
        x = self.tokenEmb(x)
        bias = x

        x = self.C_MLP_s(x)
        x = self.C_MLP_t(x)
        x = x + bias

        x = self.fc(x.transpose(1, 2).reshape(B, N, -1)).permute(0, 2, 1)
        return x