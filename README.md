# Crop Advice Snippet AI

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7rxxgwar7z/badge.svg)](https://ask.thili.ai/b/7rxxgwar7z)

This project involves building a custom GPT model from scratch to provide agricultural advice. Farmers can use a CLI tool to generate concise, context-aware advice snippets for specific crops, trained on a dataset of farming best practices. It demonstrates applying byte-pair tokenization, embeddings, causal self-attention, and transformer blocks to generate coherent, domain-specific text.

## What you are building

**Goal.** You will build a Command-Line Interface (CLI) tool that generates concise, context-aware agricultural advice snippets for specific crops, trained on a dataset of farming best practices.

**Done when.** Your CLI tool successfully generates relevant and concise agricultural advice snippets for specific crops or problems, demonstrating the ability to apply a custom-trained GPT model from scratch.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect a small corpus of agricultural advice, crop guides, or farming articles.**  
      Hint: Gather and clean raw text data from agricultural sources.
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the agricultural text.**  
      Hint: Apply the byte-pair tokenizer and create embeddings from your text corpus.
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Implement the causal self-attention mechanism and assemble the transformer block.
- [x] **4. Train the GPT model using the agricultural text data.**  
      Hint: Train your custom GPT model using the prepared agricultural text data.
- [x] **5. Implement a text generation function to provide concise advice snippets.**  
      Hint: Develop the text generation function to produce advice snippets.
- [ ] **6. Create a CLI where users can input a crop or problem and get advice.**  
      Hint: Build the command-line interface for user interaction and advice retrieval.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

Experiment with different model sizes or architectures (within the GPT framework) to observe performance changes. · Implement a mechanism to filter or rank advice snippets for relevance based on additional user input or context. · Allow users to specify the desired length or tone of the advice snippets generated.
