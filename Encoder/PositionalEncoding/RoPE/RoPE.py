import numpy as np


# 该算法是通过修改QK矩阵来引入位置编码
# 先构造旋转矩阵，再对QK矩阵应用旋转，当计算Q*K_T时，attention score 和 i−j 正相关
# ALiBi 在长度外推上非常稳定，但它是通过线性 bias 约束 attention，表达能力有限；
# RoPE 是通过旋转直接改变 QK 的几何关系，在建模复杂依赖和生成质量上更有优势，因此在大模型中更常用。

def build_rope_cache(seq_len, d_head):
    """
    构造 RoPE 所需的 sin / cos 表

    参数：
    - seq_len: 序列长度
    - d_head: 每个 attention head 的维度（必须是偶数）

    返回：
    - cos: shape = (seq_len, d_head//2)
    - sin: shape = (seq_len, d_head//2)
    """

    assert d_head % 2 == 0, "RoPE 要求 d_head 是偶数"

    # 位置索引 [0, 1, 2, ..., seq_len-1]
    position = np.arange(seq_len)[:, None]  # (seq_len, 1)

    # 频率指数：0, 2, 4, ...
    # 对应论文里的 2i / d
    dim_index = np.arange(0, d_head, 2)  # (d_head/2,)

    # 计算不同维度的角频率
    # 频率随维度指数下降（低维慢，高维快）
    inv_freq = 1.0 / (10000 ** (dim_index / d_head))

    # 每个位置、每个维度对应一个旋转角度
    # theta = position * inv_freq
    theta = position * inv_freq  # (seq_len, d_head/2)

    # 旋转用的 cos / sin
    cos = np.cos(theta)
    sin = np.sin(theta)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    对输入 x 应用 RoPE 旋转

    参数：
    - x: shape = (num_heads, seq_len, d_head)
    - cos, sin: shape = (seq_len, d_head//2)

    返回：
    - x_rot: shape = (num_heads, seq_len, d_head)
    """

    # 拆成偶数维和奇数维
    # x_even: x0, x2, x4, ...
    # x_odd:  x1, x3, x5, ...
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # RoPE 旋转公式（二维旋转）
    # [x'] = [ cos  -sin ] [x]
    # [y']   [ sin   cos ] [y]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    # 再交错拼回去
    x_rot = np.empty_like(x)
    x_rot[..., 0::2] = x_rot_even
    x_rot[..., 1::2] = x_rot_odd

    return x_rot


def attention_with_rope(Q, K, V):
    """
    带 RoPE 的 scaled dot-product attention

    Q, K, V: shape = (num_heads, seq_len, d_head)
    """

    num_heads, seq_len, d_head = Q.shape

    # 1️⃣ 构造 sin / cos
    cos, sin = build_rope_cache(seq_len, d_head)

    # 2️⃣ 对 Q / K 应用旋转
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)

    # 3️⃣ 标准 attention
    scores = np.matmul(Q_rot, K_rot.transpose(0, 2, 1)) / np.sqrt(d_head)

    # softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # 输出
    output = np.matmul(attn, V)
    return output
