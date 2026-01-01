import numpy as np


def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def longformer_attention_numpy(
        q, k, v,
        window=256,
        global_mask=None,
        attn_mask=None,
):
    """
    Longformer-style sparse attention

    q, k, v: [B, H, L, D]
    score[b, h, i, j] = q[b, h, i] · k[b, h, j]
    global_mask: [B, L] bool
    attn_mask:   [B, L] or [Batch Size, Sequence Length] bool
    return:      [B, H, L, D]
    """
    B, H, L, D = q.shape
    scale = 1.0 / np.sqrt(D)

    # global_mask是指定哪些token能够被所有token attend到
    # attn_mask是指定哪些token是padding不能被attention到的
    if global_mask is None:
        global_mask = np.zeros((B, L), dtype=bool)
    if attn_mask is None:
        attn_mask = np.ones((B, L), dtype=bool)

    # Longformer中有两类token:
    # 普通token，只看局部窗口 + 全局 token，也只有局部窗口内 token 能看它
    # 全局 token，看所有 token，所有 token 也能看它

    # ---------- 1. 局部滑动窗口注意力 ----------
    # 创建一个和Q一样形状的局部注意力输出矩阵 out_local: [B, H, L, D]
    out_local = np.zeros(q.shape, dtype=q.dtype)

    for b in range(B):
        for i in range(L):
            if not attn_mask[b, i]:
                continue

            # 获得滑动窗口的左右边界
            left = max(0, i - window)
            right = min(L, i + window + 1)
            # 根据窗口取出 qi, kj, vj
            # 原本维度: [B, H, L, D]
            qi = q[b, :, i, :]  # [H, D]
            kj = k[b, :, left:right, :]  # [H, W, D]
            vj = v[b, :, left:right, :]  # [H, W, D]

            # scores: [H, W]
            # 用每一个头的 qi 去点乘窗口内所有的 kj
            # einsum 描述:
            # 相同字母 → 要相乘并求和
            # 输出中出现的字母 → 保留为维度
            # 此处 "hd,hwd->hw" 表示:
            # h: head 维度
            # d: 特征维度
            # w: 窗口内 token 维度
            # 对于每一个 head h:
            #   对于窗口内的每一个 token w:
            #       计算 qi[h, d] 和 kj[h, w, d] 的点积，得到 scores[h, w]
            scores = np.einsum("hd,hwd->hw", qi, kj) * scale

            # padding mask
            # 使得 padding token 的 score 非常小，softmax 后接近于 0
            valid = attn_mask[b, left:right]  # [W]
            # ~表示取反，既取非 valid 的位置
            scores[:, ~valid] = -1e9

            probs = softmax(scores, axis=-1)  # [H, W]

            # out: [H, D]
            # 对于每一个 head h:
            #   对于每一个特征维度 d:
            #       计算 probs[h, w] 和 vj[h, w, d] 的加权和，得到 out[h, d]
            out_local[b, :, i, :] = np.einsum("hw,hwd->hd", probs, vj)

    # ---------- 2a. 所有 token → 全局 token ----------
    # out_all_to_global: [B, H, L, D]
    # 存储所有 token attend 到全局 token 的结果
    out_all_to_global = np.zeros_like(q)

    for b in range(B):
        g_idx = np.where(global_mask[b])[0]
        if g_idx.size == 0:
            continue

        Kg = k[b, :, g_idx, :]  # [H, G, D]
        Vg = v[b, :, g_idx, :]  # [H, G, D]

        for i in range(L):
            if not attn_mask[b, i]:
                continue

            qi = q[b, :, i, :]  # [H, D]

            # scores: [H, G]
            scores = np.einsum("hd,hgd->hg", qi, Kg) * scale

            valid_g = attn_mask[b, g_idx]
            scores[:, ~valid_g] = -1e9

            probs = softmax(scores, axis=-1)  # [H, G]

            out_all_to_global[b, :, i, :] = np.einsum(
                "hg,hgd->hd", probs, Vg
            )

    # ---------- 2b. 全局 token → 所有 token ----------
    # out_global_tokens: [B, H, L, D]
    # 存储全局 token attend 到所有 token 的结果
    out_global_tokens = np.zeros_like(q)

    for b in range(B):
        # 找出 batch b 中的全局 token 下标
        g_idx = np.where(global_mask[b])[0]
        if g_idx.size == 0:
            continue

        # 所有 token 的 K 和 V
        K_all = k[b]  # [H, L, D]
        V_all = v[b]  # [H, L, D]

        # 对每一个全局 token 进行处理
        for gi in g_idx:
            if not attn_mask[b, gi]:
                continue
            # 取出全局 token 的 Q
            qi = q[b, :, gi, :]  # [H, D]

            # scores: [H, L]
            # 用每一个头的 qi 去点乘所有 token 的 kj
            scores = np.einsum("hd,hld->hl", qi, K_all) * scale
            # padding mask
            scores[:, ~attn_mask[b]] = -1e9

            probs = softmax(scores, axis=-1)  # [H, L]
            # 计算 attention 输出
            # out: [H, D]
            # 对于每一个 head h:
            #   对于每一个特征维度 d:
            #       计算 probs[h, l] 和 V_all[h, l, d] 的加权和，得到 out[h, d]
            out_global_tokens[b, :, gi, :] = np.einsum(
                "hl,hld->hd", probs, V_all
            )

    # ---------- 3. 合并 ----------
    # 普通token：合并局部注意力和 all→global 注意力的结果
    # ===================== 3. 合并不同注意力路径的结果 =====================
    # 现在我们已经分别算出了三种 attention 输出：
    #
    # 1) out_local
    #    - 每个 token 通过「局部滑动窗口注意力」得到的表示
    #    - 所有 token 都有值（包括全局 token，但对全局 token 来说只是“临时结果”）
    #
    # 2) out_all_to_global
    #    - 每个 token 通过「token → 全局 token」得到的表示
    #    - 表示：普通 token 额外从全局 token 那里获得的信息
    #
    # 3) out_global_tokens
    #    - 只有【全局 token】的位置是有意义的
    #    - 表示：全局 token 通过「global → 所有 token」得到的表示
    #
    # 三者的 shape 全部是：
    #   [B, H, L, D]

    # ---------------------------------------------------------------------
    # 第一步：假设“所有 token 都是普通 token”
    #
    # 对普通 token 来说，最终表示应该是：
    #   局部信息（local attention）
    # + 从全局 token 获得的全局信息（token → global）
    #
    # 所以这里直接做逐元素相加：
    #   out[b, h, i, d] =
    #       out_local[b, h, i, d]
    #     + out_all_to_global[b, h, i, d]
    #
    # 注意：
    # - 这一步对【普通 token】是“最终正确结果”
    # - 但对【全局 token】只是一个“临时占位结果”，后面会被覆盖
    # ---------------------------------------------------------------------
    out = out_local + out_all_to_global

    # ---------------------------------------------------------------------
    # 第二步：修正【全局 token】的位置
    #
    # 在 Longformer 中：
    #   全局 token 只使用「global → 所有 token」的结果
    #
    # 所以：
    # - 遍历 batch 维度 b
    # - 遍历序列位置 i
    # - 如果 (b, i) 位置是全局 token
    #   → 用 out_global_tokens 中的结果，覆盖刚才的 out
    # ---------------------------------------------------------------------
    for b in range(B):
        for i in range(L):
            # global_mask[b, i] == True
            # 表示：第 b 个样本的第 i 个 token 是全局 token
            if global_mask[b, i]:
                # 用“global → all”的 attention 输出
                # 覆盖掉之前假设它是普通 token 得到的结果
                out[b, :, i, :] = out_global_tokens[b, :, i, :]

    # padding token 输出清零
    out *= attn_mask[:, None, :, None]

    return out
