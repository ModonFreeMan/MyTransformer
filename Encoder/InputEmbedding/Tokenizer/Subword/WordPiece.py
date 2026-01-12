from collections import Counter, defaultdict

# 该方法与BPE不同的是合并的策略
# 目标是最大化训练语料在当前词表下的 likelihood
# Score(w)≈P(w)/∏P(sub-pieces)
# 如果把 playing 看成一个整体
# 比拆成 play + ##ing 更能提高整体概率
# 那就加入 playing
# WordPiece 中的概率不是 token 的出现频率，而是基于一个简化语言模型的 likelihood 近似。
# 工程实现中通常用 PMI 或对数似然比来衡量“合并一个子词是否能提高整体概率”。


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
    非词首字符要加##标记，比如：play， ##ing
    并统计词频
    """
    vocab = Counter()
    for sentence in corpus:
        for word in sentence.split():
            chars = []
            for i, ch in enumerate(word):
                if i == 0:
                    chars.append(ch)
                else:
                    chars.append("##" + ch)
            chars.append("</w>")
            vocab[tuple(chars)] += 1
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
                merged = a + b.replace("##", "")
                if a.startswith("##"):
                    merged = "##" + merged.lstrip("##")
                new_word.append(merged)
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


def encode_word(word, vocab):
    tokens = []
    i = 0

    while i < len(word):
        end = len(word)
        found = None

        while end > i:
            piece = word[i:end]
            if i > 0:
                piece = "##" + piece
            if piece in vocab:
                found = piece
                break
            end -= 1

        if found is None:
            tokens.append("<unk>")
            i += 1
        else:
            tokens.append(found)
            i = end

    tokens.append("</w>")
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
