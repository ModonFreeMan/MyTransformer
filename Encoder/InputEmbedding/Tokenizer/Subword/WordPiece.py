from collections import Counter, defaultdict

CORPUS = [
    "i love natural language processing",
    "i love deep learning",
    "deep learning loves data",
    "transformers are powerful",
]

VOCAB_SIZE = 30

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def init_vocab(corpus):
    """
    将每个词拆成：字符 + </w>
    并统计词频
    """
    vocab = Counter()
    for sentence in corpus:
        for word in sentence.split():
            vocab[tuple(word) + ("</w>",)] += 1
    return vocab


def get_token_freq(vocab):
    """
    统计每个 token 在整个语料中的频率
    """
    token_freq = defaultdict(int)
    for word, freq in vocab.items():
        for token in word:
            token_freq[token] += freq
    return token_freq


def get_wordpiece_scores(vocab):
    """
    计算每个相邻 pair 的 WordPiece score
    score(a, b) = count(a, b) / (count(a) * count(b))
    """
    pair_freq = defaultdict(int)
    token_freq = get_token_freq(vocab)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq

    scores = {}
    for (a, b), freq in pair_freq.items():
        scores[(a, b)] = freq / (token_freq[a] * token_freq[b])

    return scores


def merge_pair(pair, vocab):
    """
    将 vocab 中的指定 pair 合并成一个新 token
    """
    new_vocab = Counter()
    a, b = pair

    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
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

        scores = get_wordpiece_scores(vocab)
        if not scores:
            break

        best_pair = max(scores, key=scores.get)
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

    print("=== Subword Vocab ===")
    print(subwords)

    # ② 冻结词表
    token2id, id2token = build_token2id(subwords)

    # ③ 使用 tokenizer
    print("learning →", encode("learning", merge_rules, token2id))
    print("loves →", encode("loves", merge_rules, token2id))
    print("language →", encode("language", merge_rules, token2id))
