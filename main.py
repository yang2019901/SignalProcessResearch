# verify following assertion:
#   1. attention layer is a general form of LS. The only difference is the normalizer, softmax layer.
#   ==> verification: Wq, Wk and Wv converges to L, L and In, where L is related to SVD of X
#                     after we drop the softmax layer and set loss function as mean sqaure just like LS
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb

class Attn(nn.Module):
    def __init__(self, dim_in, dim_out, normalized=False):
        super(Attn, self).__init__()
        self.Wq = nn.Linear(dim_in, dim_in, bias=False)
        self.Wk = nn.Linear(dim_in, dim_in, bias=False)
        self.Wv = nn.Linear(dim_out, dim_out, bias=False)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.normalized:bool = normalized
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-2)
    
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
        l = F.mse_loss(Y, Y_)
        return l, Y_

n, d = 1000, 3
sigma = 2
X = 20 * torch.rand((n, d), dtype=torch.float) - 10
eps = sigma * torch.randn((n, 1), dtype=torch.float)
beta = 2 + torch.randn((d, 1), dtype=torch.float)
Y = X@beta + eps
L = torch.linalg.cholesky(X.T @ X)
L_inv = L.inverse()

""" get LS estimator of beta """
beta_LS = (X.T @ X).inverse() @ X.T @ Y
Y_LS = X @ beta_LS

""" train attn """
attn = Attn(dim_in=d, dim_out=1, normalized=False)
for i in range(1000):
    loss, _ = attn.loss_LSP(X, Y)
    attn.optim.zero_grad()
    loss.backward()
    attn.optim.step()
    if i % 30:
        print(f"loss: {loss:.3f}")
torch.save(attn, f"model/attn_{loss:.3f}.mdl")

# """ load model """
# attn:Attn = torch.load("model/attn_3.682.mdl")

""" compare """
if not attn.normalized:
    Wq, Wk, Wv = attn.Wq.weight, attn.Wk.weight, attn.Wv.weight
    beta_attn_eq = Wq.T @ (X @ Wk.T).T @ Y @ Wv.T / torch.sqrt(torch.tensor(attn.dim_in))
    Y_attn = X @ beta_attn_eq.detach()
    print(beta_attn_eq.T, "\n", beta_LS.T, "\n", beta.T)
else:
    _, Y_attn = attn.loss_LSP(X, Y)

plt.scatter(X[:, 0], Y)
plt.scatter(X[:, 0], Y_LS, color="yellow")
plt.scatter(X[:, 0], Y_attn.detach(), color="orange")
plt.show()
