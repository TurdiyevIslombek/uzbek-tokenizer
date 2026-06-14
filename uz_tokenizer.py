"""
Uzbek Tokenizer (uztok)
=======================
A byte-level Byte-Pair-Encoding (BPE) tokenizer for Uzbek, with GPT-2-style
regex pre-tokenization and Uzbek-aware apostrophe normalization.

Built from scratch (no tokenizer libraries) as the tokenization component of a
from-scratch Uzbek language model. See README.md for design rationale.

Usage
-----
    from uz_tokenizer import UzbekTokenizer

    tok = UzbekTokenizer()
    tok.train(open("uz_fineweb_sample.txt", encoding="utf-8").read(), vocab_size=16384)
    tok.add_special("<|endoftext|>")
    tok.save("uz_tokenizer.json")

    ids  = tok.encode("Men oʻzbek tilini yaxshi koʻraman.")
    text = tok.decode(ids)                       # == normalized input

    tok2 = UzbekTokenizer.load("uz_tokenizer.json")
"""

import json
import regex as re
from collections import Counter


class UzbekTokenizer:
    # GPT-2 split pattern: never merge across letter / number / punctuation / whitespace fences.
    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.merges = {}      # (p0, p1) -> new_id
        self.vocab = {}       # id -> bytes
        self.specials = {}    # token string -> id
        self.compiled = re.compile(self.PATTERN)

    # ----- core BPE primitives -----
    @staticmethod
    def _get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        new_ids, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ----- Uzbek-specific normalization (lossy / canonicalizing by design) -----
    @staticmethod
    def normalize(text):
        # collapse wild apostrophe variants (curly ’ and backtick `) to plain '
        for v in ("\u2019", "`"):
            text = text.replace(v, "'")
        # o'/O'/g'/G' -> letter + okina (U+02BB), so oʻ / gʻ stay single \p{L} chunks
        for L in ("o", "O", "g", "G"):
            text = text.replace(L + "'", L + "\u02BB")
        # any remaining ' between two letters -> tutuq belgisi (U+02BC)
        return re.sub(r"(\p{L})'(\p{L})", r"\1\u02BC\2", text)

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for tok, idx in self.specials.items():
            vocab[idx] = tok.encode("utf-8")
        self.vocab = vocab

    # ----- training (dedup + frequency-weighted, with empty-stats guard) -----
    def train(self, text, vocab_size, normalize=True):
        if normalize:
            text = self.normalize(text)
        chunk_counts = Counter(self.compiled.findall(text))
        chunk_data = [(list(ch.encode("utf-8")), f) for ch, f in chunk_counts.items()]
        self.merges = {}
        for i in range(vocab_size - 256):
            stats = {}
            for ids, freq in chunk_data:
                for p, cnt in self._get_stats(ids).items():
                    stats[p] = stats.get(p, 0) + cnt * freq
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunk_data = [(self._merge(ids, pair, idx), f) for ids, f in chunk_data]
            self.merges[pair] = idx
        self._build_vocab()

    def add_special(self, token):
        idx = 256 + len(self.merges) + len(self.specials)
        self.specials[token] = idx
        self.vocab[idx] = token.encode("utf-8")

    # ----- encode / decode -----
    def _encode_chunk(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text, normalize=True):
        if normalize:
            text = self.normalize(text)
        out = []
        for ch in self.compiled.findall(text):
            out.extend(self._encode_chunk(ch))
        return out

    def decode(self, ids):
        blob = b"".join(self.vocab[i] for i in ids)
        return blob.decode("utf-8", errors="replace")

    # ----- persistence -----
    def save(self, path):
        data = {
            "pattern": self.PATTERN,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "specials": self.specials,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        tok.specials = data.get("specials", {})
        tok._build_vocab()
        return tok
