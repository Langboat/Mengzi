尽管预训练语言模型在 NLP 的各个领域里得到了广泛的应用，但是其高昂的时间和算力成本依然是一个亟需解决的问题。这要求我们在一定的算力约束下，研发出各项指标更优的模型。

我们的目标不是追求更大的模型规模，而是轻量级但更强大的模型，同时对部署和工业落地更友好。

基于语言学信息和训练加速等方法，我们研发了 Mengzi 系列模型。由于与 BERT 保持一致的模型结构，Mengzi 模型可以快速替换现有的预训练模型。

详细的技术报告请参考:

[Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](https://arxiv.org/abs/2110.06696)

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
# 下游任务
## CLUE 分数
| Model | AFQMC | TNEWS | IFLYTEK | CMNLI | WSC | CSL | CMRC2018 | C3 | CHID |
|-|-|-|-|-|-|-|-|-|-|
|RoBERTa-wwm-ext| 74.04 | 56.94 | 60.31 | 80.51 | 67.80 | 81.00 | 75.20 | 66.50 | 83.62 |
|Mengzi-BERT-base| 74.58 | 57.97 | 60.68 | 82.12 | 87.50 | 85.40 | 78.54 | 71.70 | 84.16 |

*RoBERTa-wwm-ext 的分数来自 [CLUE baseline](https://github.com/CLUEbenchmark/CLUE)*
## 对应超参
| Task | Learning rate | Batch size | Epochs |
| - | - | - | - |
| AFQMC | 3e-5 | 32 | 10 |
| TNEWS | 3e-5 | 128 | 10 |
| IFLYTEK | 3e-5 | 64 | 10 |
| CMNLI | 3e-5 | 512 | 10 |
| WSC | 8e-6 | 64 | 50 |
| CSL | 5e-5 | 128 | 5 |
| CMRC2018 | 5e-5 | 8 | 5 |
| C3 | 1e-4 | 240 | 3 |
| CHID | 5e-5 | 256 | 5 |

# 下载链接
* [Mengzi-BERT](https://huggingface.co/Langboat/mengzi-bert-base)
* [Mengzi-BERT-fin](https://huggingface.co/Langboat/mengzi-bert-base-fin)
* [Mengzi-T5](https://huggingface.co/Langboat/mengzi-t5-base)
* [Mengzi-Oscar](https://huggingface.co/Langboat/mengzi-oscar-base)

# 免责声明
该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。 实验结果可能因随机数种子，计算设备而发生改变。

# 文献引用
```
@misc{zhang2021mengzi,
      title={Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese}, 
      author={Zhuosheng Zhang and Hanqing Zhang and Keming Chen and Yuhang Guo and Jingyun Hua and Yulong Wang and Ming Zhou},
      year={2021},
      eprint={2110.06696},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
