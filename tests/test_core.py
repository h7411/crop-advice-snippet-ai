"""Tests for Crop Advice Snippet AI milestones."""
import pytest
import torch

from app.core import (
    AgriculturalCorpus, BPETokenizer, GPTEmbeddings,
    CausalSelfAttention, TransformerBlock, GPTModel,
    generate_snippet, get_agricultural_advice,
    milestone_1, milestone_2, milestone_3,
    milestone_4, milestone_5, milestone_6,
)

def test_milestone_1():
    corpus = milestone_1()
    assert isinstance(corpus, AgriculturalCorpus)
    assert len(corpus.documents) >= 5
    stats = corpus.get_stats()
    assert stats["num_words"] > 100
    assert "wheat" in corpus.cleaned_text.lower()
    assert "tomato" in corpus.cleaned_text.lower()
    assert "rice" in corpus.cleaned_text.lower()

def test_milestone_2():
    tokenizer, embeddings = milestone_2()
    assert isinstance(tokenizer, BPETokenizer)
    assert len(tokenizer.vocab) > 20
    assert len(tokenizer.merges) > 0
    sample_text = "Wheat needs soil"
    encoded = tokenizer.encode(sample_text)
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert "Wheat" in decoded
    idx = torch.tensor([encoded[:5]], dtype=torch.long)
    emb = embeddings(idx)
    assert emb.shape == (1, 5, embeddings.tok_embed.embedding_dim)

def test_milestone_3():
    block = milestone_3()
    assert isinstance(block, TransformerBlock)
    x = torch.randn(2, 8, block.attn.d_model)
    out = block(x)
    assert out.shape == x.shape
    attn = block.attn
    assert attn.bias.shape[-1] >= 8
    assert attn.bias[0, 0, 0, 1] == 0
    assert attn.bias[0, 0, 1, 0] == 1

def test_milestone_4():
    model, losses = milestone_4()
    assert isinstance(model, GPTModel)
    assert len(losses) > 0
    x = torch.randint(0, len(model.embeddings.tok_embed.weight), (2, 8))
    y = torch.randint(0, len(model.embeddings.tok_embed.weight), (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, len(model.embeddings.tok_embed.weight))
    assert loss is not None
    assert loss.item() > 0

def test_milestone_5():
    snippet = milestone_5("Wheat rust")
    assert isinstance(snippet, str)
    assert len(snippet) > 0

@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
