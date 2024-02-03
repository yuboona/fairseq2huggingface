# porting fairseq to huggingface

At a initial stage：20240203
## 1、load model

```python
from fairseq.models.transformer import TransformerModel
model = TransformerModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path=<your data path>,
  bpe=...,
  bpe_codes=...
)
```

transformers

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128
```

## 2、prepare the vocabulary

bpe？or  单字符

## 