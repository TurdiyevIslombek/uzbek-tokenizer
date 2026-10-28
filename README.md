---
language: uz
license: apache-2.0
library_name: tokenizers
tags:
  - tokenizer
  - bpe
  - uzbek
  - low-resource
---

# Uzbek Tokenizer (`uztok`)

> **This is the early development repository of the tokenizer.** The released tokenizer
> is [IslombekT/uzbek-bpe-16k](https://huggingface.co/IslombekT/uzbek-bpe-16k) on Hugging Face
> (the same `tokenizer.json` as here), and the model, paper and evaluation are in
> [uzbek-gpt-from-scratch](https://github.com/TurdiyevIslombek/uzbek-gpt-from-scratch).
> The numbers below are an early measurement on a different sample. The final measurement,
> on the 474 held-out test documents in their original web spelling, is **1.749 tokens per
> word** for this tokenizer against 2.578 for GPT-4o's `o200k_base` (Table 1 of the paper).
> The final model built on it has 103M parameters, not the 30–60M planned below.

A **byte-level BPE tokenizer for Uzbek**, built from scratch (no tokenizer
libraries), with GPT-2-style regex pre-tokenization and Uzbek-aware apostrophe
normalization. It is the tokenization component of a from-scratch Uzbek language
model and is designed to out-compress English-centric multilingual tokenizers on
Uzbek text.

## What it does

Text → integer token IDs → text, via the standard pipeline:

```
text → normalize → regex split into chunks → byte-level BPE per chunk → IDs
```

## Design decisions

1. **Byte-level BPE (256-byte base).** Every token decomposes to UTF-8 bytes, so
   there are no out-of-vocabulary characters — any Unicode (any script, emoji)
   round-trips. Vocabulary is the 256 bytes plus learned merges.

2. **GPT-2 regex pre-tokenization.** Text is split into letter / number /
   punctuation / whitespace chunks *before* BPE, so merges never fuse a word with
   adjacent punctuation (avoiding many near-duplicate `word`/`word.`/`word!`
   tokens). Letters use `\p{L}`, which covers Uzbek in **both** Latin and Cyrillic.

3. **Uzbek-aware apostrophe normalization** (deliberately lossy / canonicalizing):
   - Wild variants — ASCII `'` (U+0027), curly `’` (U+2019), backtick `` ` `` —
     are funneled to a single form.
   - `o'`, `O'`, `g'`, `G'` → letter + **okina** `ʻ` (U+02BB), so `oʻ` / `gʻ`
     remain single `\p{L}` chunks instead of shattering.
   - A remaining apostrophe **between two letters** → **tutuq belgisi** `ʼ`
     (U+02BC), e.g. `aʼlo`, `sanʼat`.
   - Apostrophes at word edges (quotation marks) are left untouched.

   *Rationale:* raw Uzbek web text mixes all three apostrophe characters
   interchangeably; without normalization the same words fragment three different
   ways, scattering common stems across junk tokens. Normalization is one-way
   (you cannot recover which character was typed) — this is intended.

4. **Vocabulary size: 16,384.** A balanced choice for a small (~30–60M parameter)
   from-scratch model on a limited Uzbek corpus: enough merges for good
   compression, few enough that token embeddings are not under-trained.

5. **Special token:** a single `<|endoftext|>` document delimiter, assigned the
   next free ID after the merges. It is inserted only by packing code, never
   produced from raw text (so stray `<|endoftext|>` strings in web data cannot
   inject false document boundaries).

## Usage

```python
from uz_tokenizer import UzbekTokenizer

tok = UzbekTokenizer()
tok.train(open("uz_fineweb_sample.txt", encoding="utf-8").read(), vocab_size=16384)
tok.add_special("<|endoftext|>")
tok.save("uz_tokenizer.json")

ids  = tok.encode("Men oʻzbek tilini yaxshi koʻraman.")
text = tok.decode(ids)                 # == normalized input

# packing documents for LM pretraining:
EOT = tok.specials["<|endoftext|>"]
stream = tok.encode(doc1) + [EOT] + tok.encode(doc2) + [EOT] + ...

tok2 = UzbekTokenizer.load("uz_tokenizer.json")
```

## Results

Trained on a FineWeb-2 `uzn_Latn` sample (90% train / 10% held-out).
Fertility = mean tokens per whitespace-delimited word on the held-out split
(lower is better).

| Tokenizer | Vocab | Fertility (tokens/word) on held-out Uzbek |
|---|---|---|
| This tokenizer | 16,384 | **1.850** |
| GPT-4o `o200k_base` | ~200,000 | 2.620 |

Measured on a 615,028-word held-out split (10% of a 50 MB FineWeb-2 `uzn_Latn`
sample, never seen in training). **The monolingual 16k tokenizer is 1.42× more
token-efficient on Uzbek than GPT-4o's ~200k multilingual tokenizer** — a win
achieved *despite* a roughly 12× smaller vocabulary, isolating the benefit of
language-specific tokenization (vocabulary size, if anything, favors GPT-4o).

## Provenance

- **Training data:** Hugging Face `HuggingFaceFW/fineweb-2`, config `uzn_Latn`
  (Uzbek Northern, Latin script), streamed sample.
- **Algorithm:** byte-level BPE (Sennrich et al. 2016; GPT-2 byte-level variant).
- **Implementation:** a from-scratch pure-Python reference implementation (this
  repo's `UzbekTokenizer`) was written to validate the algorithm end to end; the
  16k production tokenizer was then trained with the Rust-backed `tokenizers`
  library (same byte-level BPE, same GPT-2 pre-tokenization) behind this repo's
  Uzbek normalization layer, since pure-Python training is impractical at this
  vocabulary size.

## Limitations / notes

- The training loop is a **pure-Python reference implementation** — clear, but
  O(merges × unique-chunks). It is practical to ~16k vocab on a ~10 MB sample
  (tens of minutes). For the full corpus or larger vocabularies, use an
  incremental-count trainer or a fast library (`tokenizers`, `sentencepiece`) —
  the algorithm is identical.
- Normalization is intentionally lossy (see design decision 3).
- `decode(encode(x))` returns the **normalized** form of `x`, not necessarily the
  raw bytes of `x`.

## License

Apache-2.0 (see `LICENSE`), the same as the released tokenizer on Hugging Face.
