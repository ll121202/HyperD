import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .STFE import stfe
from .attention import SelfAttention
from .loss import dual_view_loss
from argparse import Namespace


class Hybrid_Periodic_Pattern(nn.Module):
    def __init__(self, period_len, num_nodes, adj, init_npy_path):
        super(Hybrid_Periodic_Pattern, self).__init__()
        self.period_len = period_len
        self.num_nodes = num_nodes
        self.A = adj

        self.linear1 = nn.Linear(num_nodes, num_nodes)
        self.linear2 = nn.Linear(2*num_nodes, num_nodes)

        self.attention_t = SelfAttention(num_nodes)
        self.attention_s = SelfAttention(period_len)

        self.init_cycle(init_npy_path)

    def init_cycle(self, init_npy_path=None):
        init_data = np.load(init_npy_path)
        assert init_data.shape == (self.period_len, self.num_nodes), \
                f"Expected shape {(self.period_len, self.num_nodes)}, but got {init_data.shape}"
        self.data = nn.Parameter(torch.from_numpy(init_data).float(), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.period_len

        data = self.data
        data = torch.einsum('ln,nv->lv', (data, self.A.to(data.device)))
        data = self.linear1(data)
        data = F.relu(data)

        data_t = self.attention_t(data)
        data_s = self.attention_s(data.transpose(0, 1)).transpose(0, 1)

        data = self.linear2(torch.cat([data_t, data_s], dim=-1))

        return data[gather_index.long()]


class HyperD(nn.Module):
    def __init__(self, **model_args):
        super(HyperD, self).__init__()
        configs = Namespace(**model_args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.num_nodes
        self.path_daily = configs.init_path_daily
        self.path_weekly = configs.init_path_weekly
        self.adj = configs.adj
        self.alpha = configs.alpha
        self.F_low = configs.F_low
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.fc_hidden_size= configs.fc_hidden_size
        self.daily_len = configs.time_of_day_size
        self.weekly_len = configs.day_of_week_size * self.daily_len

        self.daily_emb = Hybrid_Periodic_Pattern(period_len=self.daily_len, num_nodes=self.num_nodes, adj=self.adj,
                                                 init_npy_path=self.path_daily)
        self.weekly_emb = Hybrid_Periodic_Pattern(period_len=self.weekly_len, num_nodes=self.num_nodes, adj=self.adj,
                                                  init_npy_path=self.path_weekly)

        self.stfe = stfe(self.num_nodes, self.seq_len, self.pred_len, self.embed_size, self.hidden_size, self.fc_hidden_size)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):

        x = history_data[..., 0]

        index_daily = history_data[..., 1] * self.daily_len
        index_daily = index_daily[:, -1, 0]
        index_weekly = history_data[..., 1] * self.daily_len + history_data[..., 2] * self.weekly_len
        index_weekly = index_weekly[:, -1, 0]

        S_D_in = self.daily_emb(index_daily, self.seq_len)
        S_W_in = self.weekly_emb(index_weekly, self.seq_len)
        S_in = S_D_in + S_W_in

        residual_in = x - S_in
        residual_out = self.stfe(residual_in)

        S_D_out = self.daily_emb((index_daily + self.seq_len) % self.daily_len, self.pred_len)
        S_W_out = self.weekly_emb((index_weekly + self.seq_len) % self.weekly_len, self.pred_len)
        S_out = S_D_out + S_W_out
        y = residual_out + S_out

        loss_low_y, loss_high_y = dual_view_loss(y, S_out, residual_out, self.F_low)
        loss = (loss_low_y + loss_high_y) * self.alpha
        return {"prediction": y.unsqueeze(-1), "dual_view_loss": loss}

