# output=x+Sublayer(LN(x))
# x → LN → Sublayer → Add
# 梯度可直接通过残差流动
# 更容易训练上百层
import numpy as np


#  x
#  ↓
# LayerNorm(x)
#  ↓
# Sublayer
#  ↓
# x + Sublayer(LN(x))   ← 残差直接绕过 LN
#  ↓
# 输出

class LayerNorm:
    def __init__(self, hidden_dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(hidden_dim)
        self.beta = np.zeros(hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class FeedForward:
    def __init__(self, hidden_dim, ff_dim):
        self.W1 = np.random.randn(hidden_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)

        self.W2 = np.random.randn(ff_dim, hidden_dim) * 0.02
        self.b2 = np.zeros(hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        h = np.matmul(x, self.W1) + self.b1
        h = np.maximum(0, h)  # ReLU（示意）
        out = np.matmul(h, self.W2) + self.b2
        return out


class PreLNBlock:
    def __init__(self, hidden_dim, ff_dim):
        self.ln = LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ff_dim)

    def forward(self, x):
        """
        Pre-LN:
        y = x + Sublayer(LN(x))
        """
        x_norm = self.ln.forward(x)  # ① Norm
        sublayer_out = self.ffn.forward(x_norm)  # ② 子层
        out = x + sublayer_out  # ③ Add（残差）
        return out


if __name__ == '__main__':
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    ff_dim = 32

    x = np.random.randn(batch_size, seq_len, hidden_dim)

    block = PreLNBlock(hidden_dim, ff_dim)
    y = block.forward(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
