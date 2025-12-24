import numpy as np


# 该相对距离分桶算法与ALiBi相似，最终修改的是attention score，不做 position embedding
# 核心思想：位置不该是 token 的属性，而应该是「token 与 token 之间关系」的属性。当在位置 i，看位置 j 的 token 时，这个距离值对注意力应该加多少 bias？


# 核心算法，将相对位置映射到桶 id
def t5_relative_position_bucket(
        relative_position,
        num_buckets=32,
        max_distance=128,
        bidirectional=True
):
    """
    relative_position: shape = (seq_len, seq_len), 值 = j - i
    return: bucket ids, shape = (seq_len, seq_len)
    """
    ret = np.zeros_like(relative_position, dtype=np.int64)
    # 双向，一半桶给正向，一半桶给负向，使得模型能区分前后文
    if bidirectional:
        half = num_buckets // 2
        # 左侧距离：bucket 偏移 = 0
        # 右侧距离：bucket 偏移 = half
        ret += (relative_position > 0).astype(np.int64) * half
        relative_position = np.abs(relative_position)
        num_buckets = half
    else:
        # 单向（decoder / causal attention）
        # relative_position = j - i
        # j > i（未来）会被 mask 掉
        # 所以这里只保留「向左看的距离」
        #   -1, -2, -3 ...
        # 转成非负数距离：0, 1, 2 ...
        relative_position = -np.minimum(relative_position, 0)
    # max_exact 表示「精确」分配桶的最大距离
    # 小距离 → 精细，一一映射
    # 长距离 → 模糊，对数压缩
    max_exact = num_buckets // 2
    # 小于 max_exact 的距离，直接分配到精确桶
    is_small = relative_position < max_exact
    # 大于 max_exact 的距离，用 log 把距离「挤」到有限几个桶里
    large_pos = max_exact + (
        # 先把距离缩放到 [1, max_distance/max_exact]
        np.log(relative_position / max_exact + 1e-6)
        # 再归一化到 [0, 1]
        / np.log(max_distance / max_exact)
        # 再拉伸到「剩余 bucket 数」
        * (num_buckets - max_exact)
    ).astype(np.int64)
    # 防止极端情况越界
    large_pos = np.minimum(large_pos, num_buckets - 1)
    # 合并小距离和大距离的桶 id
    ret += np.where(is_small, relative_position, large_pos)
    return ret


# 根据输入序列长度和 attention head 数量，构造 T5 相对位置 bias
def build_t5_relative_bias(
        seq_len: int,
        num_heads: int,
        num_buckets=32,
        max_distance=128):
    """
    返回：
    - t5_bias: shape = (num_heads, seq_len, seq_len)
    """

    # 相对位置 j - i
    i = np.arange(seq_len).reshape(seq_len, 1)
    j = np.arange(seq_len).reshape(1, seq_len)
    relative_position = j - i

    # 分桶
    buckets = t5_relative_position_bucket(
        relative_position,
        num_buckets=num_buckets,
        max_distance=max_distance,
        bidirectional=True
    )  # (seq_len, seq_len)

    # 每个 head 一张 bucket → bias 表
    relative_attention_bias = np.random.randn(num_heads, num_buckets)

    # 查表
    t5_bias = relative_attention_bias[:, buckets]  # (heads, seq, seq)

    return t5_bias


def attention_with_t5_bias(Q, K, V, t5_bias):
    """
    Q, K, V: shape = (num_heads, seq_len, d_k)
    t5_bias: shape = (num_heads, seq_len, seq_len)
    """

    d_k = Q.shape[-1]

    # 标准 attention score
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # 加 T5 relative bias
    scores = scores + t5_bias

    # softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    # 输出
    output = np.matmul(attn, V)
    return output
