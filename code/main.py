# verify following assertion:
#   1. attention layer is a general form of LS. The only difference is the normalizer, softmax layer.
#   ==> verification: Wq, Wk and Wv converges to L, L and In, where L is related to SVD of X
#                     after we drop the softmax layer and set loss function as mean sqaure just like LS
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# for debug
# import pdb

torch.manual_seed(0)

class Attn(nn.Module):
    def __init__(self, dim_in, dim_out, normalized=False, lr=1e-3):
        super(Attn, self).__init__()
        self.Wq = nn.Linear(dim_in, dim_in, bias=False)
        self.Wk = nn.Linear(dim_in, dim_in, bias=False)
        self.Wv = nn.Linear(dim_out, dim_out, bias=False)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.normalized:bool = normalized
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    
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


def plot_shaded_curve(losses, iters, label):
    # 计算平均损失和标准差
    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)

    # 绘制平均损失曲线
    line, = plt.plot(iters, mean_losses, label=label)
    # 添加阴影部分
    plt.fill_between(iters, mean_losses-std_losses, mean_losses+std_losses, color=line.get_color(), alpha=0.2)


def main():
    def train_attn(attn:Attn, X, Y):
        li_loss = []
        for i in range(1000):
            loss, _ = attn.loss_LSP(X, Y)
            attn.optim.zero_grad()
            loss.backward()
            attn.optim.step()
            if i % 50 == 0:
                # print(f"loss: {loss:.3f}")
                li_loss.append((i, loss.item()))
        torch.save(attn, f"model/attn_{loss:.3f}.mdl")
        return list(zip(*li_loss))

    def get_attn_eq(X, Y, attn:Attn):
        Wq, Wk, Wv = attn.Wq.weight, attn.Wk.weight, attn.Wv.weight
        beta_attn_eq = Wq.T @ (X @ Wk.T).T @ Y @ Wv.T / torch.sqrt(torch.tensor(attn.dim_in))
        beta_attn_eq = beta_attn_eq.detach()
        return beta_attn_eq

    n, d = 100, 10
    sigma = 1

    losses = []
    losses_norm = []
    iters = []
    errs = []
    for _ in range(10):
        """ generate data """
        X = 20 * torch.rand((n, d), dtype=torch.float) - 10
        eps = sigma * torch.randn((n, 1), dtype=torch.float)
        beta = torch.randn((d, 1), dtype=torch.float)
        Y = X @ beta + eps

        """ define model """
        attn = Attn(dim_in=d, dim_out=1, normalized=False, lr=8e-4)
        attn_norm = Attn(dim_in=d, dim_out=1, normalized=True, lr=5e-3)

        """ train model """
        loss1 = train_attn(attn, X, Y)
        loss2 = train_attn(attn_norm, X, Y)

        beta_attn_eq = get_attn_eq(X, Y, attn)
        beta_LS = (X.T @ X).inverse() @ X.T @ Y
        print(f"beta_attn_eq: {beta_attn_eq}")
        print(f"beta_LS: {beta_LS}")
        print(f"beta: {beta}")
        print(f"dist of beta: {torch.dist(beta_LS, beta_attn_eq)}")

        iters = loss1[0]
        losses.append(loss1[1])
        losses_norm.append(loss2[1])

        errs.append(torch.dist(beta_LS, beta_attn_eq).item() / torch.norm(beta_LS).item())

    print("----------evaluation----------")

    plt.figure()
    plot_shaded_curve(losses, iters, "attn($h=id$)")
    plot_shaded_curve(losses_norm, iters, "attn($h=softmax$)")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.legend()
    plt.title(f"loss curve (different normalization, $n={n}, d={d}$)")

    plt.figure()
    bars = plt.bar(range(len(errs)), errs)
    plt.bar_label(bars, fmt='%.2g')
    plt.xlabel("experiment")
    plt.ylabel("error rate")
    plt.title(r"error rate of $\hat{\beta}_{eq}$ and $\hat{\beta}_{LS}$")

    plt.show()
main()