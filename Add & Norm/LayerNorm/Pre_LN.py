# output=x+Sublayer(LN(x))
# x → LN → Sublayer → Add
# 梯度可直接通过残差流动
# 更容易训练上百层
