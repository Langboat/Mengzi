尽管预训练语言模型在 NLP 的各个领域里得到了广泛的应用，但是其高昂的时间和算力成本依然是一个亟需解决的问题。这要求我们在一定的算力约束下，研发出各项指标更优的模型。

我们的目标不是追求更大的模型规模，而是轻量级但更强大的模型，同时对部署和工业落地更友好。

基于语言学信息和训练加速等方法，我们研发了 Mengzi 系列模型。由于保持模型结构和 BERT 一致，Mengzi 模型可以快速替换现有的预训练模型。

详细的技术报告请参考: http://www.example.com

# 导航
* [快速上手](#快速上手)
* [依赖安装](#依赖安装)
* [下载链接](#下载链接)
* [文献引用](#文献引用)

# 快速上手
## Mengzi-BERT
```python
# 使用 Huggingface transformers 加载
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")

```
## Mengzi-T5
```python
# 使用 Huggingface transformers 加载
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
```

## Mengzi-Oscar
[参考文档](https://github.com/Langboat/Mengzi/blob/main/Mengzi-Oscar.md)

# 依赖安装
```bash
pip install transformers
```

# 下载链接
* [Mengzi-BERT](https://huggingface.co/Langboat/mengzi-bert-base)
* [Mengzi-T5](https://huggingface.co/Langboat/mengzi-t5-base)
* [Mengzi-Oscar](https://huggingface.co/Langboat/mengzi-oscar-base)

# 文献引用
