# verify following assertion:
#   1. attention layer is a general form of LS. The only difference is the normalizer, softmax layer.
#   ==> verification: Wq, Wk and Wv converges to L, L and In, where L is related to SVD of X
#                     after we drop the softmax layer and set loss function as mean sqaure just like LS
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb

# torch.manual_seed(89101659224000)

class Attn(nn.Module):
    def __init__(self, dim_in, dim_out, normalized=False):
        super(Attn, self).__init__()
        self.Wq = nn.Linear(dim_in, dim_in, bias=False)
        self.Wk = nn.Linear(dim_in, dim_in, bias=False)
        self.Wv = nn.Linear(dim_out, dim_out, bias=False)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.normalized:bool = normalized
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)
    
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        """ Q: (n, din); K: (n, din); V: (n, dout) """
        assert Q.dim() == 2
        din, dout = self.dim_in, self.dim_out
        n = Q.shape[0]
        Q = self.Wq(Q).T  # (din, n)
        K = self.Wk(K).T  # (din, n)
        V = self.Wv(V).T  # (dout, n)
        weights = Q.T @ K / torch.sqrt(torch.tensor(din))
        if self.normalized:
            weights = F.softmax(weights, dim=-1)
        logits = weights @ V.T # (n, dout)
        assert logits.shape == (n, dout)
        return logits

    """ get model loss of Least Sqaure Problem """
    def loss_LSP(self, X:torch.Tensor, Y:torch.Tensor):
        X = X.view(-1, self.dim_in)
        Y = Y.view(-1, self.dim_out)
        Y_ = self.forward(X, X, Y)
        l = torch.mean(F.mse_loss(Y, Y_))
        return l

n, d = 1000, 1
sigma = 2
X = 20 * torch.rand((n, d), dtype=torch.float) - 10
eps = sigma * torch.randn((n, 1), dtype=torch.float)
beta = 2 + torch.randn((d, 1), dtype=torch.float)
Y = X@beta + eps
L = torch.linalg.cholesky(X.T @ X)
L_inv = L.inverse()

""" get LS estimator of beta """
beta_LS = (X.T @ X).inverse() @ X.T @ Y
# plt.scatter(X, Y)
# plt.plot(X, X@beta_LS)
# plt.show()

""" train attn """
attn = Attn(d, 1)
for i in range(1000):
    loss = attn.loss_LSP(X, Y)
    attn.optim.zero_grad()
    loss.backward()
    attn.optim.step()
    if i % 30:
        print(f"loss: {loss:.3f}")
torch.save(attn, f"attn_{loss:.3f}.mdl")