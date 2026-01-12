import numpy as np

# 该算法通过修改 attention score 来引入位置编码信息
# 最终是要为attention_score的每一个位置添加一个偏置，b(i,j)=−斜率⋅∣i−j∣
# 斜率的分布呈指数分布

def get_alibi_slopes(num_heads: int) -> np.ndarray:
    """
    生成 ALiBi 中每个 attention head 的 slope（斜率）

    参数：
    - num_heads: attention head 数量

    返回：
    - slopes: shape = (num_heads,)
    """

    # 使用论文/开源实现中常用的指数衰减形式
    # head 越靠前，斜率越大（越短视）
    slopes = np.array([
        2 ** (-8 * h / num_heads)
        for h in range(num_heads)
    ])

    return slopes


def build_alibi_bias(seq_len: int, num_heads: int) -> np.ndarray:
    """
    构造 ALiBi attention bias

    返回：
    - alibi_bias: shape = (num_heads, seq_len, seq_len)
    """

    slopes = get_alibi_slopes(num_heads)

    # 位置索引
    # i: query position
    # j: key position
    i = np.arange(seq_len).reshape(seq_len, 1)
    j = np.arange(seq_len).reshape(1, seq_len)

    # 相对距离 (i - j)
    # 在 causal LM 中，j > i 的部分会被 mask 掉
    relative_distance = i - j

    # 扩展到每个 head
    alibi_bias = slopes[:, None, None] * relative_distance[None, :, :]

    return alibi_bias


def attention_with_alibi(Q, K, V, alibi_bias):
    """
    带 ALiBi 的 scaled dot-product attention（简化版）

    Q, K, V: shape = (num_heads, seq_len, d_k)
    alibi_bias: shape = (num_heads, seq_len, seq_len)
    """

    d_k = Q.shape[-1]

    # 标准 attention score
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # 加上 ALiBi 偏置
    scores = scores - alibi_bias

    # softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # 加权求和
    output = np.matmul(attn, V)

    return output
