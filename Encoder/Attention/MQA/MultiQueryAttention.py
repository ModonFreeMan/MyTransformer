import numpy as np


# 主要是为了节省KV-Cache，保留多个Query，共享一个KV矩阵


def split_head(X, num_heads):
    B, S, D = X.shape
    d_k = D // num_heads
    X = X.reshape(B, S, num_heads, d_k)
    return np.transpose(X, (0, 2, 1, 3))


def split_kv_head(X):
    """
    MQA: K/V 只有一个 head
    X: (B, S, D)
    return: (B, 1, S, d_k)
    """
    B, S, D = X.shape
    return X.reshape(B, 1, S, D)


def get_attention(Qh, Kh):
    B, H, S, d_k = Qh.shape
    Kh_T = np.transpose(Kh, (0, 1, 3, 2))
    return Qh @ Kh_T / np.sqrt(d_k)


def get_mask(scores):
    B, H, S, d_k = scores.shape
    ones = np.ones((S, S), dtype=np.float64)
    tril = np.tril(ones)
    mask = tril[None, None, :, :]
    # where会自动广播
    masked_scores = np.where(mask == 0, -np.inf, scores)
    return masked_scores


def softmax(scores):
    # scores维度 B,H,Sq,Sk # 对于每一个 query，它在所有 key 上的权重之和 = 1
    # 所以针对每一个query，在其每一行也就是Sk上做softmax
    m = np.max(scores, axis=3, keepdims=True)
    e = np.exp(scores - m)
    s = np.sum(e, axis=3, keepdims=True)
    return e / s


def get_attention_output(X, Wq, Wk, Wv, Wo, num_heads):
    # ===== Q K V =====
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # ===== split =====
    Qh = split_head(Q, num_heads)  # (B,H,S,d_k)
    Kh = split_kv_head(K)  # (B,1,S,d_k)
    Vh = split_kv_head(V)  # (B,1,S,d_k)

    # ===== attention =====
    scores = get_attention(Qh, Kh)

    masked_scores = get_mask(scores)

    softmax_scores = softmax(masked_scores)

    # (B,H,S,d_k) = (B,H,S,S) @ (B,1,S,d_k)
    # 这里由于共享KV，Vh 的 head 维度是1，numpy 会自动广播，因此可以直接相乘
    attention_h = softmax_scores @ Vh

    B, S, D = X.shape
    attention = np.transpose(attention_h, (0, 2, 1, 3)).reshape(B, S, D)

    # ===== output proj =====
    output = attention @ Wo

    return output
