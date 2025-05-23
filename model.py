import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_yh = nn.Parameter(torch.randn(hidden_size, output_size))
        self.W_bh = nn.Parameter(torch.zeros(hidden_size))
        self.W_by = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_yh)

        self.f = nn.Tanh()
        
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        ht_pre = x.new_zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            xt = x[:, t, :]
            out = xt @ self.W_xh + ht_pre @ self.W_hh + self.W_bh
            ht = self.f(out)
            ht_pre = ht 
        y_pred = ht @ self.W_yh + self.W_by
        return y_pred

if __name__ == "__main__":
    model = RNN(28, 10)