import numpy as np


# 本质上是引入了一个信息瓶颈层Z，把 K/V 的信息压缩到 latent 空间中，通过 Z 获取 X 中的全局信息，然后从 Z 生成 K/V 最后计算 attention
def split_head(X, num_heads):
    """
    把 (B, S, D) 切成多头：
    (B, S, D) -> (B, H, S, d_k)
    """
    B, S, D = X.shape
    d_k = D // num_heads
    X = X.reshape(B, S, num_heads, d_k)
    return np.transpose(X, (0, 2, 1, 3))


def get_attention(Qh, Kh):
    """
    计算 attention scores:
    Qh: (B, H, Sq, d_k)
    Kh: (B, H, Sk, d_k)
    return: (B, H, Sq, Sk)
    """
    B, H, Sq, d_k = Qh.shape
    Kh_T = np.transpose(Kh, (0, 1, 3, 2))  # (B,H,d_k,Sk)
    return (Qh @ Kh_T) / np.sqrt(d_k)


def get_mask(scores):
    """
    causal mask（下三角），用于自回归训练/全序列推理时防止看未来
    scores: (B, H, S, S)
    """
    B, H, Sq, Sk = scores.shape
    ones = np.ones((Sq, Sk), dtype=np.float64)
    tril = np.tril(ones)
    mask = tril[None, None, :, :]  # (1,1,S,S) -> broadcast
    return np.where(mask == 0, -np.inf, scores)


def softmax(scores):
    """
    对最后一维（key 维度）做 softmax
    scores: (B, H, Sq, Sk)
    """
    m = np.max(scores, axis=3, keepdims=True)
    e = np.exp(scores - m)
    s = np.sum(e, axis=3, keepdims=True)
    return e / s


def get_attention_output_mla(X, Wq, W_latent, Wk_latent, Wv_latent, Wo, num_heads):
    """
    MLA: Multi-Head Latent Attention (NumPy)

    输入:
      X: (B, S, D)

    权重（推荐这样设定，最贴近你 MHA 的写法）:
      Wq:        (D, D)             # Q 从 X 直接投影得到
      W_latent:  (D, d_latent)      # 把 X 压缩到 latent
      Wk_latent: (d_latent, D)      # 从 latent 生成 K（再 split 成多头）
      Wv_latent: (d_latent, D)      # 从 latent 生成 V（再 split 成多头）
      Wo:        (D, D)             # 输出投影

    输出:
      output: (B, S, D)
    """

    # ===== 1) Q 来自原始 X（保持高分辨率 query）=====
    Q = X @ Wq  # (B,S,D)

    # ===== 2) X -> latent（信息瓶颈）=====
    Z = X @ W_latent  # (B,S,d_latent)

    # ===== 3) K/V 来自 latent（低分辨率记忆）=====
    K = Z @ Wk_latent  # (B,S,D)
    V = Z @ Wv_latent  # (B,S,D)

    # ===== 4) split heads =====
    Qh = split_head(Q, num_heads)  # (B,H,S,d_k)
    Kh = split_head(K, num_heads)  # (B,H,S,d_k)
    Vh = split_head(V, num_heads)  # (B,H,S,d_k)

    # ===== 5) attention scores =====
    scores = get_attention(Qh, Kh)  # (B,H,S,S)

    # ===== 6) causal mask =====
    masked_scores = get_mask(scores)

    # ===== 7) softmax weights =====
    attn_w = softmax(masked_scores)  # (B,H,S,S)

    # ===== 8) attention output per head =====
    attention_h = attn_w @ Vh  # (B,H,S,d_k)

    # ===== 9) concat heads =====
    B, S, D = X.shape
    attention = np.transpose(attention_h, (0, 2, 1, 3)).reshape(B, S, D)

    # ===== 10) output projection =====
    output = attention @ Wo  # (B,S,D)
    return output


if __name__ == "__main__":
    # ===== 一个最小可运行 demo =====
    np.random.seed(0)

    B, S, D = 2, 4, 8
    H = 2
    d_latent = 3  # latent 维度：越小压缩越强

    X = np.random.randn(B, S, D).astype(np.float64)

    Wq = np.random.randn(D, D).astype(np.float64)
    W_latent = np.random.randn(D, d_latent).astype(np.float64)
    Wk_latent = np.random.randn(d_latent, D).astype(np.float64)
    Wv_latent = np.random.randn(d_latent, D).astype(np.float64)
    Wo = np.random.randn(D, D).astype(np.float64)

    Y = get_attention_output_mla(X, Wq, W_latent, Wk_latent, Wv_latent, Wo, H)
    print("Y shape:", Y.shape)  # (B,S,D)
