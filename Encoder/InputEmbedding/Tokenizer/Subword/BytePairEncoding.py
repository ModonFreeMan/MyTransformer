from collections import Counter, defaultdict

CORPUS = [
    "i love natural language processing",
    "i love deep learning",
    "deep learning loves data",
    "transformers are powerful",
]

VOCAB_SIZE = 30

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# 每个词拆成字符，并加一个 词尾符号 </w>
def init_vocab(corpus):
    """
    将语料中的每个单词拆成：
    字符 + </w>
    并统计频次
    """
    vocab = Counter()
    for sentence in corpus:
        for word in sentence.split():
            vocab[tuple(word) + ("</w>",)] += 1
    return vocab


# 在语料上统计相邻单元的频数
def get_pair_freq(vocab):
    """
    统计 vocab 中所有相邻 token pair 的频率
    """
    pair_freq = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
    return pair_freq


# 合并 pair， 生成新的 vocab
def merge_pair(pair, vocab):
    """
    将 vocab 中出现的指定 pair 合并成一个新 token
    """
    new_vocab = Counter()
    a, b = pair

    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            # 如果匹配到 pair，就合并
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] += freq

    return new_vocab


def train_tokenizer(corpus=CORPUS, vocab_size=VOCAB_SIZE):
    vocab = init_vocab(corpus)
    merge_rules = []

    while True:
        # 当前 subword 集合
        subwords = set()
        for word in vocab:
            subwords.update(word)

        if len(subwords) >= vocab_size:
            break

        pair_freq = get_pair_freq(vocab)
        if not pair_freq:
            break

        best_pair = max(pair_freq, key=pair_freq.get)
        merge_rules.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

    return subwords, merge_rules


def encode_word(word, merge_rules):
    tokens = list(word) + ["</w>"]

    for a, b in merge_rules:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


def build_token2id(subwords):
    vocab_list = SPECIAL_TOKENS + sorted(subwords)
    token2id = {tok: i for i, tok in enumerate(vocab_list)}
    id2token = {i: tok for tok, i in token2id.items()}
    return token2id, id2token


def encode(text, merge_rules, token2id):
    ids = [token2id["<bos>"]]

    for word in text.lower().split():
        tokens = encode_word(word, merge_rules)
        for t in tokens:
            ids.append(token2id.get(t, token2id["<unk>"]))

    ids.append(token2id["<eos>"])
    return ids


if __name__ == "__main__":
    # ① 训练 tokenizer
    subwords, merge_rules = train_tokenizer()

    # ② 冻结词表，构建 token2id
    token2id, id2token = build_token2id(subwords)

    # ③ 使用 tokenizer
    print("i love learning →", encode("i love learning", merge_rules, token2id))
    print("deep learning →", encode("deep learning", merge_rules, token2id))
