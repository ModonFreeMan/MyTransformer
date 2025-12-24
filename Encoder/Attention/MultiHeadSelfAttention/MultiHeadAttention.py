import sys
import ast
import numpy as np


def split_head(X, num_heads):
    B, S, D = X.shape
    d_k = D // num_heads
    X = X.reshape(B, S, num_heads, d_k)
    return np.transpose(X, (0, 2, 1, 3))


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
    # scores维度 B,H,Sq,Sk
    # 对于每一个 query，它在所有 key 上的权重之和 = 1
    # 所以针对每一个query，在其每一行也就是Sk上做softmax
    m = np.max(scores, axis=3, keepdims=True)
    e = np.exp(scores - m)
    s = np.sum(e, axis=3, keepdims=True)
    return e / s


def get_attention_output(X, Wq, Wk, Wv, Wo, num_heads):


    # 由 X 生成 Q/K/V
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # 分割矩阵
    Qh = split_head(Q, num_heads)
    Kh = split_head(K, num_heads)
    Vh = split_head(V, num_heads)

    # 计算注意力
    scores = get_attention(Qh, Kh)

    # 进行掩码
    masked_scores = get_mask(scores)

    # 计算注意力权重, B, H, Sq, Sk
    softmax_scores = softmax(masked_scores)

    # 计算注意力输出, B, H, Sq, d_k
    attention_h = softmax_scores @ Vh

    # 拼接多头attention输出
    B, S, D = X.shape
    attention = np.transpose(attention_h, (0, 2, 1, 3)).reshape(B, S, D)

    # 线性投影
    output = attention @ Wo

    return output


if __name__ == '__main__':
    input_str = sys.stdin.readline().strip().split(';')
    num_heads = int(input_str[0])
    X = np.array(ast.literal_eval(input_str[1]), dtype=np.float64)
    Wq = np.array(ast.literal_eval(input_str[2]), dtype=np.float64)
    Wk = np.array(ast.literal_eval(input_str[3]), dtype=np.float64)
    Wv = np.array(ast.literal_eval(input_str[4]), dtype=np.float64)
    Wo = np.array(ast.literal_eval(input_str[5]), dtype=np.float64)
    get_attention_output(X, Wq, Wk, Wv, Wo, num_heads)
