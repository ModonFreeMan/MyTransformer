import numpy as np


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    # 初始化位置编码矩阵，全 0
    pe = np.zeros((seq_len, d_model))

    # 生成位置索引 (0, 1, 2, ..., seq_len-1)，shape = (seq_len, 1)
    position = np.arange(seq_len).reshape(-1, 1)

    # 生成维度索引 (0, 1, 2, ..., d_model-1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )
    # 通过缩放可以使得低维变化快（局部）高维变化慢（全局）

    # 偶数维：sin
    pe[:, 0::2] = np.sin(position * div_term)

    # 奇数维：cos
    pe[:, 1::2] = np.cos(position * div_term)
    # 通过 sin 和 cos 的结合，任意位移可以表示为线性组合

    return pe


if __name__ == "__main__":
    seq_len = 10
    d_model = 16
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    print(pe)
