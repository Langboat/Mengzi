[**中文**](./README.md) | [**English**](./README_en.md)

# Mengzi

尽管预训练语言模型在 NLP 的各个领域里得到了广泛的应用，但是其高昂的时间和算力成本依然是一个亟需解决的问题。这要求我们在一定的算力约束下，研发出各项指标更优的模型。

我们的目标不是追求更大的模型规模，而是轻量级但更强大，同时对部署和工业落地更友好的模型。

基于语言学信息融入和训练加速等方法，我们研发了 Mengzi 系列模型。由于与 BERT 保持一致的模型结构，Mengzi 模型可以快速替换现有的预训练模型。

详细的技术报告请参考:

[Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](https://arxiv.org/abs/2110.06696)

# 导航
* [模型介绍](#模型介绍)
* [快速上手](#快速上手)
* [依赖安装](#依赖安装)
* [下载链接](#下载链接)
* [联系方式](#联系方式)
* [免责声明](#免责声明)
* [文献引用](#文献引用)

# 模型介绍
|模型|参数量|适用场景|特点|下载链接|
|-|-|-|-|-|
|Mengzi-BERT-base|110M|文本分类、实体识别、关系抽取、阅读理解等自然语言理解类任务|与 BERT 结构相同，可以直接替换现有 BERT 权重| [HuggingFace](https://s.langboat.com/hfmengzibertbase), [国内ZIP下载](https://s.langboat.com/mengzibertbase), [PaddleNLP](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-bert-base/model_state.pdparams) |
|Mengzi-BERT-base-fin|110M|金融领域的自然语言理解类任务|基于 Mengzi-BERT-base 在金融语料上训练|[HuggingFace](https://s.langboat.com/hfmengzibertbasefin), [国内ZIP下载](https://s.langboat.com/mengzibertbasefin), [PaddleNLP](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-bert-base-fin/model_state.pdparams) |
|Mengzi-T5-base|220M|适用于文案生成、新闻生成等可控文本生成任务|与 T5 结构相同，不包含下游任务，需要在特定任务上 Finetune 后使用。与 GPT 定位不同，不适合文本续写|[HuggingFace](https://s.langboat.com/hfmengzit5base)，[国内ZIP下载](https://s.langboat.com/mengzit5base, [PaddleNLP](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-t5-base/model_state.pdparams) |
|Mengzi-Oscar-base|110M|适用于图片描述、图文互检等任务|基于 Mengzi-BERT-base 的多模态模型。在百万级图文对上进行训练|[HuggingFace](https://s.langboat.com/hfmengzioscarbase)|

# 快速上手
## Mengzi-BERT
```python
# 使用 Huggingface transformers 加载
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
```
或者
```python
# 使用 PaddleNLP 加载
from paddlenlp.transformers import BertTokenizer, BertModel

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
或者
```python
# 使用 PaddleNLP 加载
from paddlenlp.transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
```

## Mengzi-Oscar
[参考文档](./Mengzi-Oscar.md)

# 依赖安装
```bash
# 使用 Huggingface transformers 加载
pip install transformers
```
或者
```bash
# 使用 PaddleNLP 加载
pip install paddlenlp
```
# 下游任务
## CLUE 分数
| Model | AFQMC | TNEWS | IFLYTEK | CMNLI | WSC | CSL | CMRC2018 | C3 | CHID |
|-|-|-|-|-|-|-|-|-|-|
|RoBERTa-wwm-ext| 74.30 | 57.51 | 60.80 | 80.70 | 67.20 | 80.67 | 77.59 | 67.06 | 83.78 |
|Mengzi-BERT-base| 74.58 | 57.97 | 60.68 | 82.12 | 87.50 | 85.40 | 78.54 | 71.70 | 84.16 |

*RoBERTa-wwm-ext 的分数来自 [CLUE baseline](https://github.com/CLUEbenchmark/CLUE)*
## 对应超参
| Task | Learning rate | Global batch size | Epochs |
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

# 联系方式

## 微信讨论群
<img src="https://user-images.githubusercontent.com/1523477/137630760-d704aa9e-41be-4bea-8213-a83f5f027982.jpg" width="200">

## 邮箱
wangyulong[at]chuangxin[dot]com

# FAQ
**Q. mengzi-bert-base 保存的模型大小是196M。  但 bert-base 的模型大小是在 389M 左右，是定义的 base 有区别，还是保存的时候，少了一些不必要的内容？**  
A: 这是因为 Mengzi-bert-base 用 FP16 训练的。

**Q. 金融预训练模型的数据来源是什么呢？**  
A: 网页爬取的金融新闻、公告、研报。

**Q. 是否有 Tensorflow 版模型？**  
A: 可以自行转换。

**Q. 能否开源 Training 代码？**  
A:  由于和内部基础设施耦合的比较紧，目前没有计划。

**Q. 如何能做到 Langboat 官方网站上文本生成 Demo 一样的效果呢？**  
A: 我们的文本生成核心模型基于 T5 架构，基础的文本生成算法可以参考 Google 的 T5 论文： https://arxiv.org/pdf/1910.10683.pdf。
我们开源的 Mengzi-T5 模型与 Google 的 T5 预训练模型架构相同是通用的预训练模型，没有专门的文本生成任务。我们的营销文案生成功能是在其之上使用大量数据进行了具体的下游任务 Finetune。而在此基础上为了达到可控生成的效果，我们又构建了一整套文本生成 Pipeline：从数据清洗、知识抽取、训练数据构建到生成质量评价。其中大部分是按照商业落地场景进行订制的：根据不同的业务需求、不同的数据形式构建不同的预训练和 Finetune 任务。这部分牵涉到比较复杂的软件架构以及具体的业务场景，我们暂时还没有进行开源。

**Q. Mengzi-T5-base 能直接 Inference 么？**  
A: 我们参考了 T5 v1.1，不包含下游任务。

**Q: 用 Huggingface Transformer 加载出错了怎么办？**  
A: 加上 `force_download=True` 试试。

**Q: Mengzi-T5-base 在做constrain generation的时候，似乎总是倾向于生成词粒度的候选，而mT5 则相反，是字粒度优先，这个是训练过程就是词粒度处理了吗？**  
A:  我们没有用 mT5 的词表，而是基于语料重新训练了 Tokenizer，包含了更多词汇。这样同样长度的文本 encode 之后 token 数会更少些，显存占用更小，训练速度更快。




# 免责声明
该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。 实验结果可能因随机数种子，计算设备而发生改变。

使用者以各种方式使用本模型（包括但不限于修改使用、直接使用、通过第三方使用）的过程中，不得以任何方式利用本模型直接或间接从事违反所属法域的法律法规、以及社会公德的行为。使用者需对自身行为负责，因使用本模型引发的一切纠纷，由使用者自行承担全部法律及连带责任。我们不承担任何法律及连带责任。

我们拥有对本免责声明的解释、修改及更新权。

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
