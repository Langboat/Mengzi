Although pre-trained language models are widely used in various fields of NLP, their high time and computing power costs are still a pressing problem. This requires us to develop models with better metrics under certain computing power constraints.

Our goal is not to pursue a larger model size, but a lightweight but more powerful model that is also more friendly to deployment and industrial landing.

Based on linguistic information and training acceleration, we have developed the Mengzi family of models. Due to the consistent model structure with BERT, Mengzi models can quickly replace existing pre-trained models.

For detailed technical reports, please refer to: http://www.example.com

# Navigation
* [Quick Start](#quick-start)
* [Dependency](#dependency)
* [Download Links](#download-links)
* [Citation](#citation)

# Quick Start
## Mengzi-BERT
```python
# Loading with Huggingface transformers
from transformers import BertTokenizer, BertModel


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

## Mengzi-Oscar
[Reference Document](https://github.com/Langboat/Mengzi/blob/main/Mengzi-Oscar.md)

# Dependency
```bash
pip install transformers
```
# Downstream tasks
## CLUE Scores
| Model | AFQMC | TNEWS | IFLYTEK | CMNLI | WSC | CSL | CMRC2018 | C3 | CHID |
|-|-|-|-|-|-|-|-|-|-|
|RoBERTa-wwm-ext| 74.04 | 56.94 | 60.31 | 80.51 | 67.80 | 81.00 | 75.20 | 66.50 | 83.62 |
|Mengzi-BERT-base| 74.58 | 57.97 | 60.68 | 82.12 | 87.50 | 85.40 | 78.54 | 71.70 | 84.16 |
## Corresponding hyperparameters
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


*The scores of RoBERTa-wwm-ext from [CLUE baseline](https://github.com/CLUEbenchmark/CLUE)*

# Download Links
* [Mengzi-BERT](https://huggingface.co/Langboat/mengzi-bert-base)
* [Mengzi-BERT-fin](https://huggingface.co/Langboat/mengzi-bert-base-fin)
* [Mengzi-T5](https://huggingface.co/Langboat/mengzi-t5-base)
* [Mengzi-Oscar](https://huggingface.co/Langboat/mengzi-oscar-base)

# Disclaimers
The contents of this project are for technical research purposes only and are not intended as a basis for any conclusive findings. Users are free to use the models as they wish within the scope of the license, but we are not responsible for direct or indirect damages resulting from the use of the contents of this project. The experimental results presented in the technical report only indicate performance with specific data sets and super-reference combinations and are not representative of the nature of the individual models. Experimental results are subject to change due to random number seeds, computing equipment.

# Citation