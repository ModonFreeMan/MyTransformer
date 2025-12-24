# 根据输入字符串转换为对应的嵌入向量
from Encoder.InputEmbedding.Tokenizer.Subword import BytePairEncoding, Unigram, WordPiece

Tokenizer = BytePairEncoding  # 可以替换为WordPiece, Unigram需要另一种逻辑

# ① 训练 tokenizer
subwords, merge_rules = Tokenizer.train_tokenizer()

# ② 冻结词表，构建 token2id
token2id, id2token = Tokenizer.build_token2id(subwords)

# ③ 使用 tokenizer
input_string = "i love learning"
encoded_ids = Tokenizer.encode(input_string, merge_rules, token2id)
print(f"Encoded IDs for '{input_string}':", encoded_ids)

# # 使用 Unigram Tokenizer
# # ① 训练 tokenizer
# prob = Unigram.train_unigram()
#
# # ② 冻结词表
# token2id, id2token = Unigram.build_token2id_from_prob(prob)
#
# # ③ 使用 tokenizer
# input_string = "i love learning"
# encoded_ids = Unigram.encode(input_string, prob, token2id)
# print(f"Encoded IDs for '{input_string}':", encoded_ids)
