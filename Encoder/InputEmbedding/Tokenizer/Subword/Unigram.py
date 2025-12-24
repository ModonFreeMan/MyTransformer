from collections import Counter
import math

CORPUS = [
    "i love natural language processing",
    "i love deep learning",
    "deep learning loves data",
    "transformers are powerful",
]

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

VOCAB_SIZE = 30
MAX_ITER = 10
RATIO = 0.2
MAX_SUB_LEN = 10


# 初始化 Unigram 词表，包含所有子串
def init_unigram_vocab(corpus, max_sub_len):
    vocab = set()
    for sentence in corpus:
        for word in sentence.split():
            word = word + "</w>"
            # 枚举 word 中的所有子串
            for i in range(len(word)):
                for j in range(i + 1, min(len(word), i + max_sub_len) + 1):
                    vocab.add(word[i:j])
    return vocab


# 初始化每个子串的概率，均匀分布
def init_probs(vocab):
    prob = {}
    p = 1.0 / len(vocab)
    for token in vocab:
        prob[token] = p
    return prob


# 计算给定单词的最佳切分
def encode_word(word, prob):
    """
    使用 Unigram 概率对单词进行最优切分
    """
    n = len(word)
    dp = [-math.inf] * (n + 1)
    back = [None] * (n + 1)
    dp[0] = 0

    for i in range(n):
        if dp[i] == -math.inf:
            continue
        for j in range(i + 1, n + 1):
            token = word[i:j]
            if token in prob:
                score = dp[i] + math.log(prob[token])
                if score > dp[j]:
                    dp[j] = score
                    back[j] = i

    tokens = []
    i = n
    while i > 0:
        j = back[i]
        tokens.append(word[j:i])
        i = j

    return tokens[::-1]


# E-Step 期望步，计算每个子串的出现次数
def expectation_step(corpus, prob):
    counts = Counter()
    for sentence in corpus:
        for word in sentence.split():
            word = word + "</w>"
            tokens = encode_word(word, prob)
            for t in tokens:
                counts[t] += 1
    return counts


# M-Step 最大化步，更新子串概率
def maximization_step(counts):
    total = sum(counts.values())
    prob = {}
    for t, c in counts.items():
        prob[t] = c / total
    return prob


# 剪枝低概率子串
def prune_vocab(prob, ratio=0.2):
    sorted_tokens = sorted(prob.items(), key=lambda x: x[1])
    remove_n = int(len(sorted_tokens) * ratio)
    new_prob = dict(sorted_tokens[remove_n:])
    return new_prob


# 训练 Unigram 模型
def train_unigram(
        corpus=CORPUS,
        vocab_size=VOCAB_SIZE,
        max_iter=MAX_ITER,
        ratio=RATIO,
        max_sub_len=MAX_SUB_LEN
):
    vocab = init_unigram_vocab(corpus, max_sub_len)
    prob = init_probs(vocab)

    for _ in range(max_iter):
        counts = expectation_step(corpus, prob)
        prob = maximization_step(counts)

        if len(prob) <= vocab_size:
            break

        prob = prune_vocab(prob, ratio)

    return prob


def encode(text, prob, token2id):
    ids = [token2id["<bos>"]]

    for word in text.lower().split():
        tokens = encode_word(word + "</w>", prob)
        for t in tokens:
            ids.append(token2id.get(t, token2id["<unk>"]))

    ids.append(token2id["<eos>"])
    return ids


def build_token2id_from_prob(prob):
    vocab_list = SPECIAL_TOKENS + sorted(prob.keys())
    token2id = {tok: i for i, tok in enumerate(vocab_list)}
    id2token = {i: tok for tok, i in token2id.items()}
    return token2id, id2token


if __name__ == "__main__":
    # ① 训练 tokenizer
    prob = train_unigram()

    print("Vocab size:", len(prob))

    # ② 冻结词表
    token2id, id2token = build_token2id_from_prob(prob)

    # ③ 使用 tokenizer
    print("learning →", encode("learning", prob, token2id))
    print("loves →", encode("loves", prob, token2id))
