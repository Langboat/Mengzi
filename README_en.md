[**中文**](./README.md) | [**English**](./README_en.md)

# Mengzi
Although pre-trained models (PLMs) have achieved remarkable improvements in a wide range of NLP tasks, they are expensive in terms of time and resources. This calls for the study of training more efficient models that consume less computation but still ensure impressive performance.

Instead of pursuing a larger scale, we are committed to developing lightweight yet more powerful models trained with equal or less computation and friendly to rapid deployment. 

Based on linguistic information integration and training acceleration methods, we have developed the family of Mengzi models. Mengzi models can quickly replace existing pre-trained models because of the same model structure as BERT.

See [Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](https://arxiv.org/abs/2110.06696) for details.

# Navigation
* [Introduction](#introduction)
* [Quick Start](#quick-start)
* [Dependency](#dependency)
* [Download Links](#download-links)
* [Contact us](#contact-us)
* [Disclaimers](#disclaimers)
* [Citation](#citation)

# Introduction
|Model|Params|Usage Scenarios|Features|
|-|-|-|-|
|Mengzi-BERT-base|110M|Natural language understanding tasks such as text classification, entity recognition, relationship extraction, and reading comprehension|Same structure as BERT, can directly replace existing BERT weights|
|Mengzi-BERT-base-fin|110M|Natural language understanding tasks in finance|Trained on financial corpus based on Mengzi-BERT-base|
|Mengzi-T5-base|220M|Suitable for copywriting generation, news generation and other conditional generation tasks|Same structure as T5, does not contain downstream tasks and needs to be used after finetuning on a specific task. Unlike GPT, not suitable for text continuation|
|Mengzi-Oscar-base|110M|Suitable for tasks such as image caption, image-text retrieval|A multimodal model based on Mengzi-BERT-base. Trained on millions of image-text pairs|

# Quick Start
## Mengzi-BERT
```python
# Loading with Huggingface transformers
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
```
or
```python
# Loading with PaddleNLP
from paddlenlp.transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
```

## Mengzi-T5
```python
# Loading with Huggingface transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
```
or
```python
# Loading with PaddleNLP
from paddlenlp.transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
```

## Mengzi-Oscar
[Document](./Mengzi-Oscar_en.md)

# Dependency
```bash
# Loading with Huggingface transformers
pip install transformers
```
or
```bash
# Loading with PaddleNLP
pip install paddlenlp
```

# Downstream tasks
## CLUE Scores
| Model | AFQMC | TNEWS | IFLYTEK | CMNLI | WSC | CSL | CMRC2018 | C3 | CHID |
|-|-|-|-|-|-|-|-|-|-|
|RoBERTa-wwm-ext| 74.30 | 57.51 | 60.80 | 80.70 | 67.20 | 80.67 | 77.59 | 67.06 | 83.78 |
|Mengzi-BERT-base| 74.58 | 57.97 | 60.68 | 82.12 | 87.50 | 85.40 | 78.54 | 71.70 | 84.16 |

*The scores of RoBERTa-wwm-ext from [CLUE baseline](https://github.com/CLUEbenchmark/CLUE)*
## Corresponding hyperparameters
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

# Download Links
* [Mengzi-BERT](https://huggingface.co/Langboat/mengzi-bert-base) | [Mengzi-BERT (PaddleNLP)](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-bert-base/model_state.pdparams)
* [Mengzi-BERT-fin](https://huggingface.co/Langboat/mengzi-bert-base-fin) | [Mengzi-BERT-fin (PaddleNLP)](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-bert-base-fin/model_state.pdparams)
* [Mengzi-T5](https://huggingface.co/Langboat/mengzi-t5-base) | [Mengzi-T5 (PaddleNLP)](https://bj.bcebos.com/paddlenlp/models/transformers/community/Langboat/mengzi-t5-base/model_state.pdparams)
* [Mengzi-Oscar](https://huggingface.co/Langboat/mengzi-oscar-base)

# Contact Us
wangyulong[at]chuangxin[dot]com

# Disclaimers
The contents of this project are for technical research purposes only and are not intended as a basis for any conclusive findings. Users are free to use the models as they wish within the scope of the license, but we are not responsible for direct or indirect damages resulting from the use of the contents of this project. The experimental results presented in the technical report only indicate performance with specific data sets and hyperparameter combinations and are not representative of the nature of the individual models. Experimental results are subject to change due to random number seeds, computing equipment.

While using the models (which include but not limited to  modified use, direct use or use through third party), users shall not use the model in any way which will violate the laws and regulations of the jurisdiction, as well as social ethics. Users shall be responsible for their own behaviors, and taking joint legal liabilities for disputes arising from the use of models. We are not responsible for any liability arising from the use of the models.

We reserve the right to interpret, modify and update this Disclaimer.

# Citation
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
