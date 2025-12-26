import numpy as np


# 在分组请求注意力（GQA）中，唯一键和值向量的数量等于超参数 G，即组的数量。

def split_head_gqa_q(Q, num_q_heads, num_kv_heads):
    """
    Q: (B,S,D) -> (B,G,group,S,d)
    先对 Q 进行分头，将 D 维度拆分为 num_q_heads 个 head，(B,S,H,d)
    然后将 head 维度按 num_kv_heads 的数量进行分组，把 H 个 head，重新分成 G 组，每组 group 个 head，(B,G,group,S,d)
    其中 G = num_kv_heads， group = num_q_heads // num_kv
    """
    B, S, D = Q.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"
    d_k = D // num_q_heads
    G = num_kv_heads
    group = num_q_heads // num_kv_heads

    Qh = Q.reshape(B, S, num_q_heads, d_k)  # (B,S,H,d)
    Qh = np.transpose(Qh, (0, 2, 1, 3))  # (B,H,S,d)
    Qg = Qh.reshape(B, G, group, S, d_k)  # (B,G,group,S,d)
    return Qg


def split_head_gqa_kv(KV, num_kv_heads):
    """
    K/V: (B,S,D) -> (B,G,S,d)
    """
    B, S, D = KV.shape
    d_k = D // num_kv_heads
    Kh = KV.reshape(B, S, num_kv_heads, d_k)  # (B,S,G,d)
    Kh = np.transpose(Kh, (0, 2, 1, 3))  # (B,G,S,d)
    return Kh


def get_attention_gqa(Qg, Kh):
    """
    Qg: (B,G,group,S,d)
    Kh: (B,G,S,d)
    return scores: (B,G,group,S,S)
    """
    d_k = Qg.shape[-1]
    # Kh_T: (B,G,d,S)
    Kh_T = np.transpose(Kh, (0, 1, 3, 2))

    # (B,G,group,S,d) @ (B,G,1,d,S) -> (B,G,group,S,S)
    # 通过在 group 维度加一个长度为1的轴让广播更明确
    scores = Qg @ Kh_T[:, :, None, :, :]
    return scores / np.sqrt(d_k)


def get_mask_gqa(scores):
    """
    scores: (B,G,group,S,S)
    """
    B, G, group, S, _ = scores.shape
    ones = np.ones((S, S), dtype=np.float64)
    tril = np.tril(ones)
    mask = tril[None, None, None, :, :]  # (1,1,1,S,S)
    return np.where(mask == 0, -np.inf, scores)


def softmax_gqa(scores):
    """
    scores: (B,G,group,S,S)
    softmax over last axis (keys)
    """
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s


def get_attention_output_gqa_grouped(X, Wq, Wk, Wv, Wo, num_q_heads, num_kv_heads):
    """
    不使用 repeat 的 GQA 实现（reshape + 分组 attention）
    X: (B,S,D)
    Wq/Wk/Wv/Wo: (D,D)
    """

    # ===== 1) Q K V =====
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # ===== 2) split（分组）=====
    Qg = split_head_gqa_q(Q, num_q_heads, num_kv_heads)  # (B,G,group,S,d)
    Kh = split_head_gqa_kv(K, num_kv_heads)  # (B,G,S,d)
    Vh = split_head_gqa_kv(V, num_kv_heads)  # (B,G,S,d)

    # ===== 3) scores =====
    scores = get_attention_gqa(Qg, Kh)  # (B,G,group,S,S)

    # ===== 4) mask + softmax =====
    masked_scores = get_mask_gqa(scores)
    attn = softmax_gqa(masked_scores)  # (B,G,group,S,S)

    # ===== 5) attention output =====
    # (B,G,group,S,S) @ (B,G,1,S,d) -> (B,G,group,S,d)
    out_g = attn @ Vh[:, :, None, :, :]  # (B,G,group,S,d)

    # ===== 6) 合并回 (B,H,S,d) 再 concat 成 (B,S,D) =====
    B, S, D = X.shape
    d_k = D // num_q_heads
    H = num_q_heads

    out_h = out_g.reshape(B, H, S, d_k)  # (B,H,S,d)
    attention = np.transpose(out_h, (0, 2, 1, 3)).reshape(B, S, D)

    # ===== 7) 输出投影 =====
    output = attention @ Wo
    return output
