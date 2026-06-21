# Module 4 — NLP Branch *(1950s – 2017)*

**Why this era exists:** Text is fundamentally different from images. Images have fixed-size grids of pixels; text has variable-length sequences of discrete symbols where **order matters** (word order changes meaning completely). The entire NLP branch is the story of building models that understand sequence and context.

---

## Why Text Is Different from Images

**Why:** You can't just flatten a sentence into a fixed-size vector and run it through an MLP — sentences have different lengths, and the meaning of a word depends on what came before it.

**What breaks:**
- **Variable length:** "Hi" and "The quick brown fox jumps over the lazy dog" have different lengths. MLPs need fixed-size inputs.
- **Order dependency:** "Dog bites man" and "Man bites dog" are the same words, completely different meanings.
- **Long-range dependencies:** "The trophy didn't fit in the suitcase because **it** was too large" — the word "it" refers to "trophy", which appeared 8 words earlier.

**How this shapes the branch:** Every model in this module is fundamentally designed to handle sequences — processing tokens one at a time while maintaining some form of "memory" of what came before.

---

## Rule-Based NLP & N-Grams: The Dawn of Text Processing

### Why
Before statistical machine learning took over, researchers attempted to handle language using explicit logical rules (regular expressions, context-free grammars). However, human language is highly ambiguous, context-dependent, and constantly evolving, making hand-crafted rules impossible to maintain. This led to probabilistic approaches like **N-Grams**.

### What
*   **Rule-Based NLP:** Processing text using hardcoded syntactic rules (e.g., if a sentence contains "not" followed by an adjective, invert the sentiment).
*   **N-Grams:** A simple statistical language model that predicts the next word in a sequence based on the occurrences of the previous $N-1$ words. 
    *   *Unigram (1-gram):* Evaluates words independently (e.g., "the", "cat", "sat").
    *   *Bigram (2-gram):* Looks at pairs of consecutive words (e.g., "the cat", "cat sat").
    *   *Trigram (3-gram):* Looks at triplets (e.g., "the cat sat").

### How
An N-gram model estimates the probability of a word $w_t$ given its history of length $N-1$ using maximum likelihood estimation (counting frequencies in a large text corpus):
$$P(w_t \mid w_{t-N+1}, \dots, w_{t-1}) = \frac{\text{Count}(w_{t-N+1}, \dots, w_{t-1}, w_t)}{\text{Count}(w_{t-N+1}, \dots, w_{t-1})}$$
While simple, N-Grams suffer from **data sparsity**: if a specific sequence of words never appeared in the training corpus, its probability is 0.0, causing predictions to fail.

---

## Bag-of-Words & TF-IDF *(BoW classical; TF-IDF 1972)*

**Why:** Before any model can process text, text must become numbers. The earliest approach: count words and ignore order entirely.

**What:**
- **Bag of Words (BoW):** Represent a document as a vector of word counts. A vocabulary of 10,000 words → each document is a 10,000-dimensional vector where each dimension is the count of that word.
- **TF-IDF (Term Frequency–Inverse Document Frequency):** Weights word counts by how unique a word is across documents. "the" appears everywhere — low weight. "photosynthesis" is rare — high weight. `TF-IDF = TF × log(N/df)` where N is total docs and df is docs containing the word.

**How:**

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# Bag of Words
bow = CountVectorizer()
X_bow = bow.fit_transform(docs)
print(bow.get_feature_names_out())  # vocabulary
print(X_bow.toarray())              # word count matrix

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(X_tfidf.toarray().round(2))
```

**Where BoW fails:** "I love this movie, it's not bad" vs "I don't love this movie, it's bad" — same words, opposite meanings. Order is completely lost.

---

## RNN — Recurrent Neural Network *(1986)*

**Why:** To handle sequences, we need a model with memory. An RNN processes tokens one at a time and passes a "hidden state" from each step to the next — the hidden state is a compressed memory of everything seen so far.

**What:** At each time step `t`, the RNN takes the current input `xₜ` and the previous hidden state `hₜ₋₁`, combines them, and produces a new hidden state `hₜ`:

`hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)`

The same weights `Wₓ` and `Wₕ` are used at every time step (weight sharing across time).

**How:**

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# Sequence: batch=1, sequence_length=5, features=10
x = torch.randn(1, 5, 10)
output, h_n = rnn(x)

print(output.shape)  # (1, 5, 20) — hidden state at every step
print(h_n.shape)     # (1, 1, 20) — final hidden state
```

---

## The RNN Long-Range Problem *(noted from ~1991)*

**Why:** Information from early in a sequence is encoded in the hidden state, then overwritten with each new time step. By the time the RNN has processed 50 tokens, the hidden state contains almost no information about token 1.

**What:** Mathematically, this is the vanishing gradient problem applied to time: backpropagating through 50 time steps means multiplying 50 weight matrices together. Gradients shrink to zero — the model can't learn long-range dependencies.

**Example:** "The author of the book that was on the shelf in the library **was** brilliant." The word "was" must agree with "author" — 12 words back. An RNN typically fails to capture this.

**How this motivated LSTM:** A new architecture with explicit "gates" that learn what to remember and what to forget.

---

## LSTM — Long Short-Term Memory *(1997)*

**Why:** Sepp Hochreiter and Jürgen Schmidhuber designed LSTM to solve the long-range problem. The key idea: instead of one hidden state that gets overwritten, have a separate **cell state** — a highway for information that runs through the sequence with minimal interference.

**What:** An LSTM cell has three gates, each a learned sigmoid function:
- **Forget gate:** How much of the old cell state to keep: `f = σ(Wf·[h, x] + bf)`
- **Input gate:** What new information to write to cell state: `i = σ(Wi·[h, x] + bi)`
- **Output gate:** What part of cell state to output as hidden state: `o = σ(Wo·[h, x] + bo)`

The cell state update: `C = f ⊙ C_prev + i ⊙ tanh(Wc·[h, x] + bc)`

**How:**

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(1, 50, 10)   # longer sequence now
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (1, 50, 20)
print(c_n.shape)     # (1, 1, 20) — cell state carries long-range memory
```

LSTMs became the dominant sequence model from ~2014 onward and are still used today where interpretability of the sequence memory matters.

---

## GRU — Gated Recurrent Unit *(2014)*

**Why:** LSTMs work well but have 4 sets of weight matrices per layer — computationally heavy. Cho et al. asked: can we get similar performance with a simpler design?

**What:** GRU merges the forget and input gates into a single **update gate**, and merges the cell state and hidden state. Two gates instead of three, no separate cell state.

**How:**

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(1, 50, 10)
output, h_n = gru(x)
print(output.shape)  # (1, 50, 20)
```

GRU trains faster than LSTM and performs comparably on many tasks. The choice between them is empirical — try both.

---

## Word Embeddings: Word2Vec, GloVe & FastText

### Why
One-hot encoding represents words as sparse, high-dimensional vectors (e.g. $[0, 0, 1, 0, \dots]$) where the vector length is equal to the vocabulary size. Under this representation, every word is orthogonal to all others: "cat" and "kitten" share zero mathematical similarity. We need dense, low-dimensional vectors (embeddings) that capture semantic relationships, where words used in similar contexts reside close to each other in vector space.

---

### Word2Vec: Predictive Local Windows *(2013, Mikolov et al.)*

#### What
Word2Vec learns word representations by training a single-layer neural network on a predictive task. It assumes that "you shall know a word by the company it keeps" (Distributional Hypothesis).
*   **Continuous Bag of Words (CBOW):** Predicts a target word given its surrounding context words.
*   **Skip-gram:** Predicts the surrounding context words given a single target word.

#### How
Word2Vec slides a context window of size $C$ across a corpus. For Skip-gram, it maximizes the probability of context words $w_{t+j}$ given the center word $w_t$:
$$\mathcal{L} = \sum_{t=1}^{T} \sum_{-C \le j \le C, j \neq 0} \log P(w_{t+j} \mid w_t)$$
To train efficiently, it uses **Negative Sampling**, converting the multi-class classification (predicting a word out of a vocabulary of $V$ items) into a binary classification problem (predicting whether a word pair is a real context pair or a random noise pair).

```
   CBOW:      [Context Words] ───► [Average / Sum] ───► [Predict Target Word]
   Skip-gram: [Target Word]  ─────────────────────────► [Predict Context Words]
```

#### Code
Here is the code to train a Word2Vec model using Gensim:
```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "land"],
    ["man", "and", "woman", "are", "humans"],
]

# Train model
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, epochs=100)

# Retrieve similarity
similarity = model.wv.similarity("king", "queen")
print(f"Similarity (king, queen): {similarity:.3f}")
```

---

### GloVe: Global Co-occurrence Factorization *(2014, Pennington et al.)*

#### Why
Word2Vec only looks at local context windows (e.g., 5 words at a time), failing to leverage the global co-occurrence statistics of the entire corpus. 

#### What
**GloVe (Global Vectors for Word Representation)** combines the advantages of local context window methods (like Word2Vec) and global matrix factorization methods (like Latent Semantic Analysis). It constructs a global word-word co-occurrence matrix $X$, where $X_{ij}$ is the number of times word $j$ appears in the context of word $i$.

#### How
GloVe minimizes a weighted least-squares objective function that directly fits the log-probabilities of word co-occurrences:
$$\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
Where:
*   $w_i$ and $\tilde{w}_j$ are the word vectors.
*   $b_i$ and $\tilde{b}_j$ are biases.
*   $f(X_{ij}) = (X_{ij} / x_{\max})^\alpha$ is a weighting function that prevents very frequent co-occurrences (like "the sat") from dominating the loss.

---

### FastText: Subword Character N-Grams *(2016, Bojanowski et al.)*

#### Why
Word2Vec and GloVe assign a single vector to each unique word. Consequently:
1.  **Out-of-Vocabulary (OOV) Words:** If a word was not seen during training (e.g. "transformerless"), the model cannot generate a vector for it.
2.  **Morphology:** Words sharing roots (e.g. "run", "running", "ran") are treated as entirely independent, ignoring morphological structure.

#### What
Developed by Facebook AI Research, **FastText** represents each word as a bag of **character n-grams**. For example, for the word "where" and $n=3$, the character n-grams are:
$$\text{<wh}, \text{whe}, \text{her}, \text{ere}, \text{re>}$$
along with the special word token `<where>`.

#### How
The word vector for "where" is the **sum of the vectors of its character n-grams**. 
*   *Handling OOV:* If a new word is encountered, its vector is constructed by summing the vectors of its constituent character n-grams. Even if the full word is novel, its subwords (like prefixes and suffixes) are likely familiar, letting the model represent it accurately.

---

---

## Seq2Seq / Encoder-Decoder *(2014)*

**Why:** Tasks like translation, summarization, and question answering require mapping a variable-length input sequence to a variable-length output sequence. A single RNN can't do this — you'd need to know the output length in advance.

**What:** Two separate RNNs (or LSTMs): an **encoder** reads the entire input sequence and compresses it into a fixed-size vector (the context vector). A **decoder** takes that vector and generates the output sequence one token at a time.

**How:**

```
Input:  "The cat sat"  →  Encoder  →  context vector  →  Decoder  →  "Le chat s'assit"
```

```python
# Conceptual structure — PyTorch seq2seq
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        return self.lstm(self.embed(x))  # returns (output, (h_n, c_n))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden
```

**The bottleneck problem:** The entire input sequence must be compressed into one fixed-size vector. For long inputs, this loses information — the decoder can't "look back" at specific parts of the input. This motivated attention.

---

## Attention — Bahdanau *(2014)* & Luong *(2015)*

**Why:** The fixed-size context vector in Seq2Seq is a bottleneck. When translating a long sentence, the decoder needs to focus on different parts of the input at each output step — not just the final compressed vector. Attention lets the decoder look back at all encoder hidden states and decide which ones matter most for the current output token.

**What:** At each decoder step, attention computes a **score** for every encoder hidden state — how relevant is each input token for the current output token? Scores are normalized into weights (softmax), and a weighted sum of encoder states is computed. This weighted sum is the attention vector, given to the decoder as additional context.

**How:**

```python
import torch
import torch.nn.functional as F

# encoder_outputs: all hidden states from encoder [batch, src_len, hidden]
# decoder_hidden:  current decoder state [batch, 1, hidden]

def bahdanau_attention(encoder_outputs, decoder_hidden):
    # Score every encoder output against current decoder state
    scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
    # scores shape: [batch, 1, src_len]

    weights = F.softmax(scores, dim=-1)  # normalize to sum=1
    context = torch.bmm(weights, encoder_outputs)  # weighted sum
    # context shape: [batch, 1, hidden]
    return context, weights
```

The attention weights are directly interpretable — you can visualize which input words the model focused on when generating each output word. This was a major step toward explainability in NLP.

**This is the direct bridge into Module 5.** Attention bolted onto RNNs improved translation dramatically. The next logical question: what if we kept the attention mechanism and threw away the RNN entirely?

---

## NLP Evaluation Metrics

When you build an NLP model, how do you know if it's actually good? Classification tasks (sentiment, intent) can use F1 and accuracy from Module 1. But generative NLP tasks — translation, summarization — produce free-form text. You can't just count right/wrong. These metrics were designed for that problem.

---

### BLEU — Bilingual Evaluation Understudy *(2002, Papineni et al.)*

**The problem it solves:** You train a machine translation model. The model outputs "The cat is on the mat." The reference (human) translation is "The cat sat on the mat." How similar are they, automatically and at scale?

BLEU measures n-gram overlap between the generated text and one or more human references.

```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"

1-gram matches: The, cat, on, the, mat (5/6 words) = 0.83
2-gram matches: "The cat", "on the", "the mat" (3/5) = 0.60
BLEU ≈ geometric mean of n-gram precisions × brevity penalty
```

**Score ranges:** 0–100. Above 30 is considered decent for translation. State-of-the-art models reach 40–50.

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate  =  ["the", "cat", "is",  "on", "the", "mat"]

score = sentence_bleu(reference, candidate)
print(f"BLEU: {score:.3f}")  # ~0.63
```

**Failure mode:** BLEU only measures word overlap, not meaning. "The feline rested upon the carpet" and "The cat sat on the mat" mean the same thing but BLEU scores them very low. This is why BLEU has been largely supplemented (though not replaced) by semantic metrics like BERTScore in modern research.

---

### ROUGE — Recall-Oriented Understudy for Gisting Evaluation *(2004, Lin)*

Where BLEU focuses on precision ("did the generated text use the right words?"), ROUGE focuses on recall ("did the generated text cover the important words from the reference?"). This makes ROUGE more natural for **summarization** — you care about coverage, not whether the summary adds extra words.

**Three variants:**
- **ROUGE-1:** Unigram (single word) recall
- **ROUGE-2:** Bigram (two-word phrase) overlap
- **ROUGE-L:** Longest Common Subsequence — captures sentence structure even without exact phrase matches

```python
# pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

reference = "The quick brown fox jumps over the lazy dog near the river"
generated = "The fox jumps over the lazy dog"

scores = scorer.score(reference, generated)
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")  # word overlap
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")  # structural similarity
```

**Failure mode:** Like BLEU, ROUGE measures surface overlap, not meaning. A high ROUGE score doesn't guarantee a useful summary — it might just copy sentences verbatim from the source.

---

### When to Use Which NLP Metric

| Task | Metric | Why |
|------|--------|-----|
| Machine translation | BLEU | Standard; enables benchmark comparisons across papers |
| Summarization | ROUGE-L | Measures coverage of key content |
| Question answering | Exact Match + F1 | Whether the answer span overlaps with the reference |
| Semantic similarity | BERTScore | Embedding-based; captures meaning not just surface overlap |
| LLM output quality | Human eval + GPT-as-judge | No automatic metric fully captures helpfulness |

> **Perplexity** (how surprised the language model is by test text) is the core intrinsic metric for language models — covered in Module 6 where LLMs are introduced.

---

## Quick Reference & Common Questions — NLP

### The Big Picture in Plain English

Language is different from images. "Dog bites man" and "Man bites dog" use the exact same words but mean completely different things — **order matters**. "He went to the bank to fish" and "He went to the bank to deposit money" — **context matters**. Classical ML treats text as a bag of words (order doesn't matter). That's wrong for language understanding.

**The progression:**
- Bag of Words: ignore order → works for simple classification
- RNN/LSTM: process one word at a time, remember context → works for longer sequences
- Attention: look at all words simultaneously, focus on relevant ones → Transformers (next module)

---

### Module 4 Q&A

**Q: What is the difference between BoW and TF-IDF?**
A: BoW counts word occurrences — "the" appears 50 times gets weight 50. TF-IDF also accounts for how common a word is across all documents. "The" appears in every document → very low IDF → nearly zero TF-IDF weight. "neuroscience" appears in few documents → high IDF → high TF-IDF weight. TF-IDF highlights the words that are distinctive to a document.

**Q: Why do RNNs struggle with long sequences?**
A: RNNs pass information through a fixed-size hidden state vector. As sequences get longer, older information gets overwritten — the hidden state is a bottleneck. Also, vanishing/exploding gradients during backpropagation through time (BPTT) make training on long sequences unstable. LSTMs and GRUs partially address this with gating mechanisms that control what to remember/forget.

**Q: What is the difference between LSTM and GRU?**
A: LSTM has 3 gates (input, forget, output) and a separate cell state. GRU has 2 gates (reset, update) and no separate cell state. GRU is simpler and trains faster; LSTM is more expressive. In practice, performance is similar — GRU is a good default for many sequence tasks, LSTM slightly better on complex long sequences.

**Q: What are word embeddings and why are they better than one-hot encoding?**
A: One-hot encoding represents each word as a vector with one 1 and all other values 0. "King" and "queen" have zero similarity. Word2Vec/GloVe embeddings represent words as dense vectors (e.g., 300 dimensions) where semantically similar words are close: "king" - "man" + "woman" ≈ "queen". Embeddings capture meaning, analogy, and relationships that one-hot can't.

**Q: What is seq2seq and what is it used for?**
A: Sequence-to-sequence models (Sutskever et al. 2014) encode an input sequence into a fixed vector (encoder), then decode it into a different sequence (decoder). Used for: machine translation (English → French), text summarization (article → summary), chatbots (question → answer). The attention mechanism (Bahdanau 2015) extended seq2seq by allowing the decoder to attend to all encoder states at each step, dramatically improving quality on long sequences.

---