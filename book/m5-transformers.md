# Module 5 — The Convergence: Transformers *(2017 – 2020)*

**Why this era exists:** By 2016, two entirely separate engineering disciplines — the computer vision track (CNNs from Module 3) and the NLP track (LSTMs from Module 4) — had quietly been running in parallel for years. Vision teams processed images spatially; language teams processed words sequentially. They used different architectures, attended different conferences, and read different papers. In 2017, a single eight-page paper wiped out that division. It introduced one mechanism — **attention** — that worked for language, vision, audio, protein sequences, and virtually everything else. That paper was called *"Attention Is All You Need."* This module is the story of how it happened and how it works.

---

## The Problem: The Serial Processing Prison

Picture a factory assembly line where every worker can only start after the one before them finishes. Worker 2 can't touch the widget until Worker 1 is done. Worker 100 can't start until Workers 1 through 99 have each completed their turn. Now imagine you have 10,000 workers available, all sitting idle, waiting for their one moment in the chain.

That was an LSTM processing a sentence.

LSTMs and RNNs are fundamentally sequential: to compute the hidden state for word 50, you must first compute the hidden state for words 1 through 49, in that exact order. Modern GPUs are designed to run thousands of operations *simultaneously* — but an LSTM's dependency chain meant that 99% of those parallel cores sat unused. Training a single language model on a large dataset took weeks.

The second problem was memory. Even with the gating mechanisms of LSTMs, information still had to travel through a long corridor of multiplications. In a 500-word document, the meaning of the opening sentence had to be compressed into a vector, passed through 490 intermediate operations, and somehow survive intact to influence the understanding of the closing sentence. It rarely did. Long-range dependencies — the kind that connect a pronoun on page 3 to the noun it refers to on page 1 — were where LSTMs quietly failed.

By 2016, researchers had already bolted attention *onto* LSTMs (the Bahdanau attention from Module 4) and seen quality improvements. Attention allowed the decoder to look directly at any encoder state rather than relying on the compressed bottleneck. It worked. But then someone asked the obvious question.

---

## The Breakthrough Question

*"If attention is doing all the important work of connecting distant words — why do we still need the LSTM at all?"*

This question was asked by a team at Google Brain in 2016. If attention can directly relate any word to any other word, then the recurrent connections are not connecting anything new — they're just dead weight that forces sequential processing. What if you built a model out of nothing but attention, stacked many times?

They tried it. The recurrent layers were removed entirely. What remained was a stack of attention operations and simple feedforward networks, processing the entire sequence in one parallel pass. Training time for state-of-the-art machine translation dropped from weeks to hours. The model was better. The paper, published in June 2017, was titled *"Attention Is All You Need"* by Vaswani et al.

The Transformer was born.

---

## Self-Attention: The Library Research Desk

Before the full architecture, you need to understand the core mechanism that powers everything: **self-attention**.

### Why Self-Attention

In previous models, each word only saw nearby words (CNN-style) or only saw the past (RNN-style). Self-attention says: every word gets to look at every other word in the sentence simultaneously and decide how much context to borrow from each.

Consider the sentence: *"The animal didn't cross the street because **it** was too tired."*

What does "it" refer to? "Animal" or "street"? You know immediately it's "animal" — because tired applies to animals, not streets. But for a machine, resolving this requires comparing "it" to every other word and finding the highest relevance. Self-attention does exactly that: it computes a relationship score between "it" and every word in the sentence, finds that "animal" scores highest, and pulls information from "animal" into the representation of "it."

If the sentence were *"...because **it** was too wide,"* attention would pull from "street" instead. The same mechanism, working correctly from context.

### The Library Research Desk Analogy

Think of self-attention like a library research desk.

You walk up to the desk with a **research question** — that's your **Query (Q)**. You're looking for information relevant to a specific topic.

The library has a **card catalogue** — a list of every book's title, subject, and keywords. That's the **Key (K)** for each book. The keys describe what each book contains.

When your query matches a key well, you pull the actual book and read it — that's the **Value (V)**. The values are the actual content you absorb.

The model does this mathematically:
1. For every word, create three vectors — a Query (what am I looking for?), a Key (what do I contain?), a Value (what information do I carry?).
2. Compare the Query of the current word against the Keys of every other word (including itself).
3. The better the match, the higher the attention score.
4. Use those scores to compute a weighted sum of all Values — absorbing more from highly relevant words and less from irrelevant ones.

```
  Word "it" → [Query Q_it]
                   │
                   ▼
   ┌──────────────────────────────────────────┐
   │  Compare Q_it against K for every word:  │
   │    K_animal → high similarity score      │
   │    K_street → low similarity score       │
   │    K_tired  → medium similarity score    │
   └──────────────────────────────────────────┘
                   │
                   ▼  Softmax → weights sum to 1
   ┌──────────────────────────────────────────┐
   │  Weighted sum of Values:                 │
   │    0.7 × V_animal + 0.1 × V_street + ... │
   └──────────────────────────────────────────┘
                   │
                   ▼
         Enriched representation of "it"
         (now "understands" it = animal)
```

### The Math

For a full sequence of tokens, all queries, keys, and values are packed into matrices Q, K, and V. The computation becomes one matrix operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **$QK^T$** — dot products between every query and every key. A large value = high similarity.
- **$\sqrt{d_k}$** — we divide by the square root of the key dimension. Without this, as dimensions grow large, dot products grow huge, the softmax saturates to near-zero gradients, and learning stalls.
- **Softmax** — converts scores into probabilities (attention weights that sum to 1).
- **Multiply by V** — the weighted blend of values becomes the new, context-enriched representation.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)  # Dimension of key vectors

    # Step 1: Compute raw similarity between every Q-K pair
    # Q: [batch, seq_len, d_k] | K^T: [batch, d_k, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: (Optional) Causal mask — hide future tokens for decoders
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # -inf → softmax → 0

    # Step 3: Softmax turns scores into weights that sum to 1
    weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted blend of values = enriched representation
    output = torch.matmul(weights, V)
    return output, weights
```

### What Self-Attention Cannot Do

Self-attention has no concept of word order. The formula $\text{softmax}(QK^T/\sqrt{d_k})V$ is **permutation-invariant**: if you shuffle the words, the attention weights shuffle too, but the *set* of outputs is identical. "Dog bites man" and "Man bites dog" would produce the same result. This is why positional encoding (covered later) is essential.

Self-attention also cannot easily capture strictly local patterns. A CNN naturally knows that adjacent pixels are more likely to be related than distant ones. Self-attention has no such prior — it must *learn* locality from data. This requires more data and more training, which is why early Transformers needed much larger datasets than CNNs.

---

## The Full Transformer Block: End-to-End

Self-attention is the *heart* of a Transformer, but a Transformer block has more parts. Understanding the complete block is the most important interview concept in this module.

Here is a single Transformer block — the unit that gets stacked 12 times in BERT, 96 times in GPT-3, and 120 times in GPT-4:

```
         [Input Token Embeddings + Positional Encoding]
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Multi-Head         │
                  │  Self-Attention     │
                  └─────────────────────┘
                            │
                            ▼
                    [Add & Layer Norm]  ← residual: output + input
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Feed-Forward       │
                  │  Network (FFN)      │
                  │  (2 linear layers   │
                  │   + activation)     │
                  └─────────────────────┘
                            │
                            ▼
                    [Add & Layer Norm]  ← residual: output + input
                            │
                            ▼
                  [Output — passed to next block]
```

### Part 1: Multi-Head Attention
Covered in the next section. Multiple attention heads run in parallel so the block can track several relationship types at once.

### Part 2: Add & Layer Norm
After the attention output is computed, it is added back to the original input before the attention layer (*residual connection*, same idea as ResNet from Module 3), and then passed through Layer Normalization.

Why the residual? Training very deep networks risks vanishing gradients — the same problem that broke deep MLPs before ResNet. The shortcut gives gradients a direct path backward, stabilizing training.

Why Layer Norm? It normalizes each token's vector independently (mean ≈ 0, variance ≈ 1), preventing any single dimension from growing unboundedly large and destabilizing the attention scores.

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN: expand to d_ff (typically 4× d_model) then compress back
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # activation — smooth ReLU variant
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sub-layer 1: Multi-Head Attention + residual + norm
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))   # Add & Norm

        # Sub-layer 2: Feed-Forward Network + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))     # Add & Norm
        return x
```

### Part 3: The Feed-Forward Network (FFN)

The FFN is often overlooked but critical. It runs independently on each token position after attention:

- **Expand**: Linear layer projects from $d_{model}$ (e.g., 512) to $d_{ff}$ (e.g., 2048) — typically 4× wider.
- **Activate**: A non-linearity (GELU or ReLU) — without this, two linear layers are still just one linear layer.
- **Compress**: Linear layer projects back down to $d_{model}$.

Why does the FFN exist if attention already enriched the representation? Attention is essentially a routing mechanism — it decides *which* tokens to pull information from. But it operates linearly (weighted sums). The FFN is where non-linear transformations happen — where the model learns to actually *process* the routed information. Research has shown that the FFN layers store and recall factual knowledge ("Paris is the capital of France"). Attention decides *where* to look; FFN decides *what to do* with what it found.

---

## Multi-Head Attention: The Panel of Experts

### Why One Head Fails

If you run attention only once per block, the model can only focus on one type of relationship at a time. For "She threw the ball to her brother at the park," a single attention pass might correctly link "She" → "threw." But you simultaneously need: "threw" → "ball" (action-object), "ball" → "brother" (recipient), "park" → "threw" (location of action). A single head has to split its attention budget across all of these and does all of them weakly.

### Eight Librarians, Eight Research Questions

Multi-head attention runs $H$ independent attention operations in parallel, on smaller slices of the embedding. Think of hiring 8 specialist librarians: one focuses on subject-verb links, one on pronoun resolution, one on prepositional phrases, one on named entities. Each specialist examines the sentence through a different lens.

Technically: the $d_{model}$-dimensional embedding is split into $H$ chunks of size $d_k = d_{model}/H$. Each head gets its own learned $W_Q$, $W_K$, $W_V$ projection matrices, runs independent attention, and produces its own output. All $H$ outputs are concatenated and projected back to $d_{model}$ by a final weight matrix $W_O$.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Each head has its own Q, K, V projection (packed into one matrix)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # final merge projection

    def forward(self, x):
        B, T, D = x.shape
        H, dk = self.num_heads, self.d_k

        # Project and reshape: [Batch, Heads, SeqLen, HeadDim]
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)

        # Scaled dot-product attention — all heads computed in parallel
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)
        weights = F.softmax(scores, dim=-1)
        out = weights @ V   # [B, H, T, dk]

        # Merge heads back: [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)  # final projection
```

In practice, GPT-3 uses 96 attention heads across a 12,288-dimensional model. Each head operates on a 128-dimensional slice. The model learns that some heads track long-range subject-verb agreement, others track adjacent syntactic structure, and others learn relationships that no human has a clean linguistic name for.

---

## Positional Encoding: Giving Seats to Words

Self-attention is order-blind. The sentence "The dog bit the man" and "The man bit the dog" produce the same attention scores for the same words — just in different positions. Since word order is meaning, this is a serious problem.

The solution: before feeding tokens into the Transformer, we add a **positional encoding vector** to each token's embedding. This vector encodes the position of that token (0th, 1st, 2nd...) so the model can distinguish position 0 from position 50.

### Sinusoidal Encoding (Original, 2017)

The original Transformer used fixed sine and cosine waves of different frequencies:

$$\text{PE}_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Think of it like a musical chord. Each dimension of the positional vector vibrates at a different frequency. Low-frequency dimensions distinguish broad position (beginning vs. end of document). High-frequency dimensions distinguish fine position (word 10 vs. word 11). Together, every position gets a unique "fingerprint" that the model can learn to read.

The clever mathematical property: the encoding of position $pos + k$ can always be expressed as a linear combination of the encoding of position $pos$. This means the model can generalise to relative distances between words without needing to memorise every absolute position.

### RoPE: Rotary Position Embeddings (2022)

Sinusoidal encoding was elegant but had limitations: it added positional information *before* the attention computation, but attention itself was still position-agnostic internally. In 2022, **RoPE (Rotary Position Embeddings)** introduced a better approach: inject position *into the attention operation itself*, by rotating the Q and K vectors before computing their dot products.

The key insight: if you rotate the Query and Key vectors of two positions by angles proportional to their positions, their dot product naturally encodes the *relative distance* between them, not just absolute position. A word at position 5 attending to a word at position 2 "knows" they are 3 steps apart — regardless of where in a long document they appear.

RoPE became the standard for all modern LLMs: LLaMA, Mistral, GPT-4, Gemini. Unlike learned absolute embeddings, RoPE generalises naturally to sequence lengths longer than those seen during training — a key advantage for models with 128K+ token context windows.

---

## The BERT vs. GPT Fork: Two Camps, Two Goals

Once the full Transformer was established, researchers had a choice: which half of the architecture should a language model use? The encoder reads everything bidirectionally; the decoder generates autoregressively. In 2018, two teams made opposite choices — and both were right for different reasons.

```
              [Full Transformer — 2017]
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
   [Encoder-Only]              [Decoder-Only]
      BERT (2018)                GPT (2018)
      Google Brain               OpenAI
  Bidirectional reading    Left-to-right generation
  "Understand the text"    "Complete the text"
           │
           └─────────────┐
                          ▼
                  [Encoder-Decoder]
                     T5 (2019)
                      Google
                "Translate the task"
```

### BERT: The Detective

**BERT (Bidirectional Encoder Representations from Transformers)** uses only the encoder. Its training task was **Masked Language Modeling**: randomly hide 15% of words (replace with `[MASK]`) and force the model to predict what they were.

The crucial word is *bidirectional*. When predicting the masked word, BERT can see all words to the left *and* right of the mask. It reads the entire sentence, in both directions, before making a decision — like a detective who reads the entire case file before naming the suspect.

This makes BERT exceptional at *understanding* tasks: classifying sentiment, extracting named entities, answering comprehension questions, powering semantic search (M6 uses BERT-style encoders extensively for document retrieval). But BERT is poor at *generating* text, because generation requires predicting tokens one at a time from left to right — something BERT was never trained to do.

```
Training input:  "The cat sat on the [MASK]."
BERT predicts:   "mat" (seeing both "cat sat on" and the period after)
```

### GPT: The Storyteller

**GPT (Generative Pre-trained Transformer)** uses only the decoder. Its training task was **Next-Token Prediction**: given any sequence of text, predict the very next word.

To prevent cheating, GPT adds a **causal mask** — an upper-triangular matrix of $-\infty$ values that, after softmax, zeroes out attention to any future token. The model can only see the past. This is why GPT-style models can write the next sentence of a story but cannot "understand" the full document the way BERT can.

What GPT gained: the ability to generate arbitrarily long text, one token at a time, coherently. GPT-3 in 2020 stunned researchers by performing tasks it was never explicitly trained for — writing code, answering trivia, translating languages — purely from patterns learned during next-token prediction at massive scale.

```
Training input:  "The cat sat on the"
GPT predicts:    "mat" (only seeing left context)
Then appends "mat" → predicts next → "."
```

### T5: The Universal Converter

In 2019, Google introduced **T5 (Text-to-Text Transfer Transformer)**, using the full encoder-decoder architecture. T5's insight was radical: *every NLP task can be framed as text-in, text-out*.

Translation? `"translate English to German: The house is blue"` → `"Das Haus ist blau"`. Sentiment? `"classify: I loved the film"` → `"positive"`. Summarization? `"summarize: [long article]"` → `"[one paragraph]"`. A single model, one training recipe, all tasks. T5 showed that the task format was the only thing that needed to change — the weights could stay the same.

| Architecture | Example | Bidirectional | Generates | Best For |
|:---|:---|:---|:---|:---|
| Encoder-only | BERT, RoBERTa | ✅ Yes | ❌ No | Classification, Search, NER |
| Decoder-only | GPT series, LLaMA | ❌ No | ✅ Yes | Generation, Chat, Code |
| Encoder-Decoder | T5, BART | ✅ Encoder | ✅ Decoder | Translation, Summarization |

---

## Vision Transformer (ViT): Images Are Just Sequences

By 2020, the Transformer had conquered language. The question became: could it conquer vision too?

CNNs had reigned in vision since 2012 (Module 3). They worked because of built-in assumptions: nearby pixels are related (locality), and a cat is a cat wherever it appears (translation invariance). These assumptions were baked into the convolution operation itself. They were also a constraint.

A CNN has no direct way to connect the top-left corner of an image to the bottom-right corner without passing through many layers of downsampling. A Transformer's attention, by contrast, connects every patch to every other patch in a single operation.

### The Puzzle Board

Imagine an art restorer examining a damaged painting. They don't analyze it pixel by pixel, and they don't scan it left-to-right like reading a book. They break it into meaningful regions — this corner has a sky, this patch has a hand, this section shows fabric. Then they consider how all the regions relate to each other to understand the full picture.

That is exactly what ViT does.

**Vision Transformer (ViT)** splits an image into a grid of fixed-size patches (typically 16×16 pixels). Each patch is flattened and linearly projected into an embedding vector — treated exactly like a word token. A learnable `[CLS]` token is prepended. Positional embeddings are added. Then the entire sequence is fed into a standard Transformer encoder.

```
  [224×224 Image]
       │
       ▼
  [Grid of 14×14 = 196 patches of 16×16 pixels]
       │
       ▼
  [Linear Projection: each 768-dim vector]
       │
  [+ Positional Embedding]
       │
       ▼
  [Standard Transformer Encoder — 12 blocks]
       │
       ▼
  [CLS token output → Classification head]
```

```python
class ViTPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2   # 196 patches
        # A Conv2d with kernel=patch_size, stride=patch_size projects each patch
        # to embed_dim in one efficient operation
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # [B, embed_dim, 14, 14]
        x = x.flatten(2)           # [B, embed_dim, 196]
        return x.transpose(1, 2)   # [B, 196, embed_dim]
```

### The Trade-off

Because ViT has no locality prior, it needs significantly more data to learn that adjacent patches are related — information a CNN gets for free from its convolution kernel. When trained on ImageNet alone (1.2M images), ViT underperforms ResNet. But when pretrained on JFT-300M (300 million images), ViT matches or exceeds CNNs on every benchmark.

The lesson: CNNs are more data-efficient because they have stronger assumptions. ViT scales better because it has weaker assumptions and can learn patterns that CNNs' fixed local windows cannot capture. At large enough scale, less inductive bias wins.

By 2022, ViT variants (DeiT, Swin Transformer, BEiT) had become the dominant architecture in computer vision research, completing the Transformer's conquest from NLP to vision.

---

## The Full LLM Request Lifecycle

This is one of the most common interview questions for any AI engineering role. When you type a message into ChatGPT and press Enter, exactly what happens inside the model?

```
[1] Your Prompt (raw text)
       │
       ▼
[2] Tokenizer
       │  "Hello world" → [15496, 995]
       ▼
[3] Embedding Matrix Lookup
       │  Each token ID → dense vector [d_model dimensions]
       ▼
[4] + Positional Encoding (RoPE or sinusoidal)
       │  Inject position information into each embedding
       ▼
[5] Transformer Layers × N
       │  Each layer: Multi-Head Attention → Add&Norm → FFN → Add&Norm
       │  N = 12 (GPT-2) | 32 (LLaMA-7B) | 96 (GPT-3) | ~120 (GPT-4)
       ▼
[6] Final Hidden State (last token position)
       │  A vector of d_model dimensions representing "what comes next"
       ▼
[7] Language Model Head (Linear layer)
       │  Projects d_model → vocabulary size (50,000+ tokens)
       │  Output = logits: a raw score for every possible next token
       ▼
[8] Softmax
       │  Converts logits to probabilities: each token's P(next)
       ▼
[9] Sampling Strategy
       │  Greedy: pick argmax (highest probability token)
       │  Temperature: scale logits before softmax (higher T = more random)
       │  Top-K: sample only from top K candidates
       │  Top-P (nucleus): sample from smallest set summing to probability P
       ▼
[10] New Token → append to sequence → repeat from step 2
       │  Continue until [EOS] token or max_tokens reached
       ▼
[Output] Complete response
```

One important detail: the model does **not** plan the full response before writing word one. It generates one token at a time, appending each to the input and running the full forward pass again. A 200-word response requires 200 complete forward passes through all N layers. This is why inference is expensive and why KV caching (next section) is critical.

### Why This Architecture Causes Hallucination

Understanding the lifecycle makes it obvious why LLMs hallucinate. The model never retrieves facts from a database. It has no pointer to a "truth store." At step 9, it is literally sampling from a probability distribution over the vocabulary — the same mechanism whether it is writing a correct mathematical proof or inventing a fake citation. It has no internal flag that says "I am not sure about this."

Mechanically, hallucination happens because:
1. **Distribution mismatch** — the training data is imperfect; if wrong facts appeared frequently enough, the model learned them.
2. **Overconfident completion** — next-token prediction always produces *something*. There is no option to output "I don't know" unless that was the statistically correct continuation of the prompt in training data.
3. **No working memory** — the model cannot double-check its previous sentence against an external source mid-generation.
4. **Sampling randomness** — temperature > 0 means the model deliberately introduces variation. That variation sometimes lands on an incorrect but statistically plausible token chain.

This is why RAG (Module 6) exists: it forces the model to ground generation in retrieved facts before generating, giving it an external truth source to complete against.

---

## Speed Optimizations: Making Transformers Practical

Standard self-attention has a problem at scale: it computes an $N \times N$ similarity matrix where $N$ is the sequence length. At $N = 100{,}000$ tokens (a medium-length book), that matrix has 10 *billion* entries — more than fits in any GPU's memory.

Three innovations made modern LLMs deployable.

### 1. Flash Attention: Compute in Tiles, Not in Full (2022)

The bottleneck was not the GPU's compute speed — it was the speed of *transferring data between the GPU's main memory (HBM) and its fast on-chip cache (SRAM)*. Standard attention writes the full $N \times N$ matrix to HBM, reads it back for softmax, writes the result again, reads it back for the weighted sum. At large $N$, these memory transfers dominate training time.

**Flash Attention** (Tri Dao et al., 2022) computes attention in **tiles** that fit entirely within the fast SRAM. It never materializes the full $N \times N$ matrix. The softmax is computed incrementally across tiles using a numerically stable running-max trick. The result: identical mathematical output to standard attention, but:
- Memory usage drops from $O(N^2)$ to $O(N)$
- Training speed improves 2–4× on long sequences
- Enables 32K, 128K, and 1M+ token context windows that would otherwise be impossible

Flash Attention V2 and V3 have continued to extend these improvements. Every serious LLM training run today uses Flash Attention.

### 2. KV Cache: Never Recompute What You Already Computed

During autoregressive generation, the model generates token 1, then processes tokens 1–2 to get token 3, then processes tokens 1–3 to get token 4. Without caching, each step recomputes the Key and Value matrices for all previous tokens — expensive, and completely redundant, because those tokens have not changed.

**KV Cache** saves the computed $K$ and $V$ matrices for all tokens already processed. When generating the next token, the model only computes $Q$, $K$, $V$ for the *new* token, then concatenates the new $K$ and $V$ with the cache. Attention runs once over the full cache.

```
Without KV Cache:
  Token 100 → recompute K, V for tokens 1–100 → expensive O(N²) per step

With KV Cache:
  Token 100 → load cached K, V for 1–99 → compute only K, V for token 100
              → run attention → fast O(N) per step
```

The trade-off: KV cache consumes significant GPU memory. For a GPT-3-scale model generating 4K tokens with 96 layers, the cache can easily consume tens of gigabytes. This is one of the primary memory bottlenecks in LLM serving.

### 3. MQA and GQA: Compressing the KV Cache (2019, 2023)

Standard multi-head attention assigns each of the $H$ heads its own separate $K$ and $V$ matrices. For 32 heads, that means 32 sets of Key and Value vectors to cache — memory scales linearly with heads.

**Multi-Query Attention (MQA)** (Shazeer, 2019) uses a single shared $K$ and $V$ pair for all heads. Only the $Q$ remains head-specific. This shrinks the KV cache by a factor of $H$ (e.g., 32×) with only a minor quality degradation.

**Grouped-Query Attention (GQA)** (Ainslie et al., 2023) is the practical middle ground: group the $H$ heads into $G$ groups, share $K$ and $V$ within each group. LLaMA 2 70B uses 8 KV heads for 64 query heads (8× cache reduction). Mistral, Gemini, and most modern production LLMs use GQA.

```
Standard MHA:  Q₁K₁V₁, Q₂K₂V₂, ..., Q₃₂K₃₂V₃₂  (32 KV pairs)
MQA:           Q₁K₁V₁, Q₂K₁V₁, ..., Q₃₂K₁V₁   (1 KV pair, shared)
GQA (G=8):     Heads 1–4 share K₁V₁, heads 5–8 share K₂V₂, etc. (8 KV pairs)
```

---

## Quick Reference: Module 5 Q&A

**Q: What is the Transformer Block? Walk me through it.**
**A:** Input embeddings + positional encoding → Multi-Head Self-Attention → Add & LayerNorm (residual) → Feed-Forward Network (expand–activate–compress) → Add & LayerNorm (residual) → output to next block. Stacked N times.

**Q: Why does scaling by $\sqrt{d_k}$ matter?**
**A:** Large $d_k$ makes dot products grow large in magnitude. Large dot products push softmax to near-0 or near-1 outputs where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the dot products in a stable range where softmax produces informative gradients.

**Q: Why does BERT use bidirectional attention but GPT uses causal masking?**
**A:** BERT was designed for *understanding* tasks — it reads the entire input and then classifies or extracts. Bidirectional attention gives it the full context of every word. GPT was designed for *generation* — predicting the next word. If it could see future tokens during training, it would simply copy the answer rather than learn to predict it.

**Q: What is a causal mask?**
**A:** An upper-triangular matrix of $-\infty$ values added to the attention scores before softmax. After softmax, positions beyond the current token get weight 0 — the model cannot attend to anything it hasn't "said" yet.

**Q: What did Flash Attention actually change?**
**A:** It kept the math identical but changed *how* the computation is physically executed — using GPU SRAM tiles instead of writing to slow HBM. The result is the same but 2–4× faster with dramatically less memory at long sequence lengths.

**Q: Why do modern LLMs use GQA instead of full multi-head attention?**
**A:** KV cache is a major memory bottleneck during inference. GQA reduces the number of unique Key-Value pairs from $H$ (one per head) to $G$ (one per group), cutting cache memory by $H/G$ times. LLaMA 2 70B uses 8 KV heads vs 64 query heads — an 8× cache reduction with negligible quality loss.

**Q: Why does ViT need more data than a CNN to achieve the same accuracy?**
**A:** A CNN bakes in locality (nearby pixels are related) and translation invariance (a cat anywhere = a cat). These are correct assumptions for most images, so a CNN needs fewer examples to learn good features. ViT has no such priors — it must learn locality from scratch. At small data scale, assumptions win. At massive data scale, flexibility wins.

---

## Bridge to Module 6

You now have the complete Transformer architecture: attention, multi-head attention, positional encoding, the full block, and the speed optimizations that make it deployable.

But right now, the Transformer is just an architecture — a blueprint. What happens when you train this blueprint on the entire internet, with 175 billion parameters, for months on thousands of GPUs?

You get GPT-3. And when GPT-3 was released, something unexpected happened: the model could write code, translate languages, answer trivia, and summarize documents — none of which it was explicitly trained to do. It had learned all of these as *emergent side effects* of predicting the next word at massive scale.

Module 6 is about what happens when the Transformer meets industrial-scale data and compute — and why the resulting behavior surprised even its creators.