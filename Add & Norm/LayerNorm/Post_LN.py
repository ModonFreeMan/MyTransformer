# output=LN(x+Sublayer(x))
# x → Sublayer → Add → LN
# 层数很深时训练不稳定
# 梯度路径较长
import numpy as np


#  x
#  ↓
# FFN(x)
#  ↓
# x + FFN(x)        ← Add（残差）
#  ↓
# LayerNorm        ← Norm
#  ↓
# 输出

class LayerNorm:
    def __init__(self, hidden_dim, eps=1e-5):
        self.eps = eps
        # 可学习参数
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


class PostLNBlock:
    def __init__(self, hidden_dim, ff_dim):
        self.ffn = FeedForward(hidden_dim, ff_dim)
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Post-LN:
        y = LN(x + Sublayer(x))
        """
        sublayer_out = self.ffn.forward(x)
        residual = x + sublayer_out
        out = self.ln.forward(residual)
        return out


if __name__ == '__main__':
    # 超参数
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    ff_dim = 32

    # 输入
    x = np.random.randn(batch_size, seq_len, hidden_dim)

    # Post-LN block
    block = PostLNBlock(hidden_dim, ff_dim)
    y = block.forward(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
