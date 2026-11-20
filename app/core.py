"""Crop Advice Snippet AI: Custom GPT model built from scratch for agricultural advice."""
import re
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

RAW_AGRICULTURAL_CORPUS = """
Crop Guide: Wheat Management
Wheat requires well-drained loamy soil with moderate moisture. Apply nitrogen fertilizer in split doses during tillering and flowering. Control rust and powdery mildew with fungicides at early onset. Irrigate at crown root initiation and grain filling stages for optimal yield.

Crop Guide: Rice Cultivation
Paddy rice flourishes in clayey soils with high water retention. Maintain standing water of 2 to 5 cm during vegetative and reproductive stages. Manage blast disease and stem borer using integrated pest management. Drain fields two weeks before harvest to promote uniform ripening.

Crop Guide: Tomato Care
Tomatoes demand well-aerated sandy loam soil with rich organic compost. Water deeply at the root zone twice weekly, avoiding overhead watering to prevent early blight. Support vines with stakes or trellises to improve air circulation. Apply balanced potassium and phosphorus fertilizers during fruit set.

Crop Guide: Corn and Maize
Corn needs full sunlight and nitrogen-rich soil with good drainage. Space rows 75 cm apart to allow adequate airflow and sunlight penetration. Scout for fall armyworm and corn earworm during early whorl stage. Irrigate regularly during silking and tasseling to ensure complete kernel formation.

Crop Guide: Potato Production
Plant seed potatoes in loose, well-drained soil rich in organic matter. Hill soil around stems as plants grow to protect tubers from sunlight and greening. Scout for late blight and Colorado potato beetles regularly. Stop irrigation two weeks before vine killing and harvest to mature skin.

Crop Guide: Cotton Farming
Cotton prefers deep, fertile, well-aerated black or alluvial soils. Monitor closely for bollworm and whitefly infestations using sticky traps. Optimize drip irrigation to avoid waterlogging during squaring and flowering. Apply potassium to enhance fiber strength and boll weight.

Crop Guide: Soybean Management
Inoculate soybean seeds with Rhizobium bacteria to fix atmospheric nitrogen. Maintain weed-free conditions during the first 40 days of vegetative growth. Watch for rust and pod borer attacks during pod development. Harvest when moisture drops to 13 percent to prevent pod shattering.
"""

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

class AgriculturalCorpus:
    def __init__(self, raw_text: str = RAW_AGRICULTURAL_CORPUS):
        self.raw_text = raw_text
        self.cleaned_text = clean_text(raw_text)
        self.documents = [doc.strip() for doc in self.cleaned_text.split("\n\n") if doc.strip()]

    def get_stats(self) -> Dict[str, int]:
        words = self.cleaned_text.split()
        return {
            "num_docs": len(self.documents),
            "num_words": len(words),
            "num_chars": len(self.cleaned_text),
            "vocab_estimate": len(set(words)),
        }

def milestone_1() -> AgriculturalCorpus:
    return AgriculturalCorpus()

class BPETokenizer:
    def __init__(self, vocab_size: int = 300):
        self.target_vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[int, str] = {}
        self.inverse_vocab: Dict[str, int] = {}

    def _get_stats(self, token_lists: List[List[str]]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for tokens in token_lists:
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, token_lists: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        merged_lists = []
        p0, p1 = pair
        replacement = p0 + p1
        for tokens in token_lists:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(replacement)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            merged_lists.append(new_tokens)
        return merged_lists

    def train(self, text: str) -> None:
        token_lists = [list(word) for word in text.split(" ")]
        for i in range(len(token_lists)):
            token_lists[i].append("</w>")

        unique_tokens = set(ch for tokens in token_lists for ch in tokens)
        self.vocab = {idx: tok for idx, tok in enumerate(sorted(unique_tokens))}
        self.inverse_vocab = {tok: idx for idx, tok in self.vocab.items()}

        next_idx = len(self.vocab)
        while next_idx < self.target_vocab_size:
            stats = self._get_stats(token_lists)
            if not stats:
                break
            best_pair = max(stats.items(), key=lambda item: item[1])[0]
            if stats[best_pair] < 2:
                break
            self.merges.append(best_pair)
            merged_tok = best_pair[0] + best_pair[1]
            self.vocab[next_idx] = merged_tok
            self.inverse_vocab[merged_tok] = next_idx
            next_idx += 1
            token_lists = self._merge_pair(token_lists, best_pair)

    def encode(self, text: str) -> List[int]:
        words = text.split(" ")
        ids: List[int] = []
        for word in words:
            tokens = list(word) + ["</w>"]
            for pair in self.merges:
                p0, p1 = pair
                rep = p0 + p1
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                        new_tokens.append(rep)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for t in tokens:
                if t in self.inverse_vocab:
                    ids.append(self.inverse_vocab[t])
                else:
                    for ch in t:
                        ids.append(self.inverse_vocab.get(ch, 0))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.vocab.get(idx, "") for idx in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 128):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Sequence length {t} exceeds max {self.max_seq_len}"
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
        return self.tok_embed(idx) + self.pos_embed(positions)

def milestone_2() -> Tuple[BPETokenizer, GPTEmbeddings]:
    corpus = milestone_1()
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(corpus.cleaned_text)
    embeddings = GPTEmbeddings(vocab_size=len(tokenizer.vocab), d_model=32, max_seq_len=64)
    return tokenizer, embeddings


import math
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

def milestone_3() -> TransformerBlock:
    return TransformerBlock(d_model=32, n_heads=4, block_size=64)



import random
from typing import Optional

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        block_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.embeddings = GPTEmbeddings(vocab_size, d_model, block_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def train_gpt_model(
    model: GPTModel,
    data_ids: List[int],
    steps: int = 30,
    batch_size: int = 4,
    lr: float = 3e-3,
) -> List[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: List[float] = []
    block_size = model.block_size

    if len(data_ids) <= block_size + 1:
        data_ids = data_ids * ((block_size + 2) // len(data_ids) + 1)

    model.train()
    for _ in range(steps):
        ix = [random.randint(0, len(data_ids) - block_size - 1) for _ in range(batch_size)]
        x = torch.stack([torch.tensor(data_ids[i : i + block_size], dtype=torch.long) for i in ix])
        y = torch.stack(
            [torch.tensor(data_ids[i + 1 : i + block_size + 1], dtype=torch.long) for i in ix]
        )
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def milestone_4() -> Tuple[GPTModel, List[float]]:
    tokenizer, _ = milestone_2()
    corpus = milestone_1()
    token_ids = tokenizer.encode(corpus.cleaned_text)
    model = GPTModel(vocab_size=len(tokenizer.vocab), d_model=32, n_heads=4, n_layers=2, block_size=32)
    losses = train_gpt_model(model, token_ids, steps=15, batch_size=4)
    return model, losses


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")

def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
