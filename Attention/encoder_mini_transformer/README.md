# Tiny Transformer Encoder Classifier (Learning Project)

Small PyTorch project to practice building a **Transformer encoder-only** model with **padding + attention mask** and a simple **binary classification** task on synthetic token sequences.

## Task
Given a padded sequence of token ids, predict:

- **y = 1** if the pattern **`1, 9` appears contiguously** somewhere in the *valid* (non-padding) part of the sequence.
- **y = 0** otherwise.

## What this project practices
- `nn.Embedding` for token embeddings
- positional embeddings (learned)
- `TransformerEncoderLayer` forward pass
- `src_key_padding_mask` (True = padding)
- pooling for sequence classification (masked mean in this project)
- training loop + logging

## Model (high level)
1. Token embedding + positional embedding  
2. Transformer encoder layer  
3. Pooling (masked mean / CLS)  
4. MLP head → logits `(B, 1)`  

Loss: `BCEWithLogitsLoss`

## Dataset
Synthetic sequences with variable lengths + padding.
Optionally supports **hard negatives**:
- negatives contain `1` and `9` but **never** as the contiguous bigram `1,9`
(to avoid shortcut learning like “if 1 and 9 appear → positive”).

