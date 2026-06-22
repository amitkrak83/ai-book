# Module 4 — NLP Branch *(1950s – 2017)*

**Why this era exists:** Text is fundamentally different from images. An image is a fixed-size grid of numbers — a 224×224 photo is always 50,176 numbers. Text is a variable-length sequence of discrete symbols where **order is meaning**: "Dog bites man" and "Man bites dog" use the same four words and mean completely opposite things. Every model in this module is the story of engineers building more and more sophisticated ways to handle sequence and context — until they finally solved both problems in 2014 with attention, which directly triggered the Transformer revolution of Module 5.

---

## Why Text Cannot Be an Image

Before building any NLP model, you need to feel the exact problem that makes text hard. It is not just that text is "different" — it is that three specific properties of text break every technique that worked for images.

**Problem 1 — Variable length.** "Hi" is 2 characters. "The United Nations Security Council resolution passed unanimously" is 65 characters. An MLP or CNN requires a fixed-size input. You cannot feed both of these sentences into the same model without inventing some way to handle length.

**Problem 2 — Order is meaning.** The words "I paid the doctor" and "The doctor paid me" use exactly the same four words with completely different meanings. A model that treats text as a bag of words — just counting what's there without caring about sequence — will give both sentences identical representations.

**Problem 3 — Long-range dependencies.** Consider: *"The trophy didn't fit in the suitcase because **it** was too large."* What does "it" refer to? The trophy. You need to connect a pronoun at position 12 to the noun at position 2. Classical sequence models lose track of earlier context over distance. This is the core problem the entire module is trying to solve.

Every technique below is a response to one or more of these three problems.

---

## The Bag-of-Words Era: Throwing Away Order Entirely

The earliest approach was blunt: just count the words and forget that order exists. This sounds wrong — and it is, for complex language tasks — but it was surprisingly useful for simple ones.

### Bag of Words and TF-IDF *(BoW classical; TF-IDF 1972)*

Imagine you're a librarian sorting 10,000 books into topics. You don't read each book — you empty it onto the floor, count which words appear, and pile them into a word-frequency histogram. That histogram *is* Bag of Words.

A vocabulary of 10,000 words gives every document a 10,000-dimensional vector. Entry 42 = how many times "photosynthesis" appeared. Entry 1,337 = how many times "the" appeared. You lose all word order, but you can now compare documents mathematically using cosine similarity.

**TF-IDF** fixes one obvious flaw in BoW: "the" appears 500 times in a document but tells you almost nothing about the topic. A word's weight should be high when it appears *often in this document* (Term Frequency) but *rarely across all documents* (Inverse Document Frequency). This upweights distinctive words and downweights noise.

$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \log\!\left(\frac{N}{\text{df}(w)}\right)$$

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# Bag of Words — raw counts, order ignored
bow = CountVectorizer()
X_bow = bow.fit_transform(docs)
print(bow.get_feature_names_out())   # vocabulary
print(X_bow.toarray())               # count matrix

# TF-IDF — downweight common words, upweight distinctive ones
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(X_tfidf.toarray().round(2))
```

**Where BoW breaks:** "I love this movie, it's not bad" vs "I don't love this movie, it's bad." Same words, opposite meanings. Order completely lost. This is the fundamental ceiling of the BoW approach — and what motivated every subsequent technique.

---

## N-Grams: Peeking at Word Pairs *(statistical NLP, 1950s onward)*

Before neural networks entered NLP, researchers tried to predict the next word using statistics. The idea: count how often each word sequence appears in a large corpus, then use those frequencies to predict the next word.

An **N-gram** model looks at the previous $N-1$ words to predict the current one:
- *Bigram (2-gram):* "The cat ___" → look up all times "cat" was followed by something → predict "sat"
- *Trigram (3-gram):* "The cat sat ___" → look up the full 3-word history

$$P(w_t \mid w_{t-N+1}, \dots, w_{t-1}) = \frac{\text{Count}(w_{t-N+1}, \dots, w_t)}{\text{Count}(w_{t-N+1}, \dots, w_{t-1})}$$

N-grams work for short-range patterns — "New York" almost always follows "in", "the" almost always precedes a noun. But they hit a wall called **data sparsity**: any 4-word sequence that never appeared in the training corpus gets probability zero, and your model breaks. Larger N means more context but exponentially more sparsity. N-grams were the state of the art for language modeling until neural networks arrived.

---

## RNN: The First Machine with Memory *(1986)*

The breakthrough insight that opened the neural NLP era: **what if the network had a running internal memory that persisted across words?**

Imagine you're listening to a sentence word by word, but you can't write anything down. Instead, you're holding a mental summary in your head, updating it with each new word. That mental summary is the **hidden state** of an RNN.

### How an RNN Works

At every time step $t$, the RNN takes two inputs: the current word $x_t$ and the previous hidden state $h_{t-1}$ (the memory of everything before). It combines them to produce a new hidden state $h_t$:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

The same weights $W_x$ and $W_h$ are applied at every step. The hidden state gets passed to the next step, carrying a compressed representation of the entire history.

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# Sequence: batch=1, 5 words, each a 10-dimensional embedding
x = torch.randn(1, 5, 10)
output, h_n = rnn(x)

print(output.shape)  # (1, 5, 20) — hidden state at every word position
print(h_n.shape)     # (1, 1, 20) — final summary of entire sequence
```

### Why RNNs Stall on Long Sequences *(~1991)*

The hidden state is a fixed-size vector. Each new word overwrites part of it. After 50 words, the hidden state contains almost nothing about word 1. It is like a whiteboard with finite space — writing new information gradually erases the old.

Mathematically, this is the **vanishing gradient problem applied to time**. Backpropagating through 50 time steps means multiplying 50 weight matrices together. Gradients shrink exponentially. The model cannot learn to connect a pronoun at position 50 to the noun it refers to at position 3.

Example of what breaks: *"The author of the book that was sitting on the shelf in the corner of the library **was** brilliant."* The verb "was" must agree with "author" — 15 words back. A plain RNN almost always fails to make this connection.

---

## LSTM: The Long-Term Memory Highway *(1997)*

In 1997, Sepp Hochreiter and Jürgen Schmidhuber published the paper that would define a decade of NLP. Their insight was surgical: the problem is not that RNNs have memory — it is that they have *only one type* of memory. When you want to remember something from 30 words ago, you're fighting against the constant updates that accumulate from the 30 intermediate words.

The solution: give the LSTM **two separate memory channels**:
1. **Hidden state** ($h_t$) — a short-term working memory that updates every step
2. **Cell state** ($C_t$) — a long-term memory highway that carries information across many steps with minimal interference

The cell state is protected by three **gates** — learned sigmoid functions that output values between 0 (block) and 1 (pass). Think of them as valves:

- **Forget gate** ($f_t$): How much of the old cell state to keep? *"Should I forget that the subject was singular?"*
- **Input gate** ($i_t$): What new information to write into the cell state? *"The subject just changed to plural."*
- **Output gate** ($o_t$): What part of the cell state to expose as the hidden state? *"Output the relevant piece for predicting the next word."*

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate new memory)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(update cell state)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(output hidden state)}$$

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(1, 50, 10)  # now a 50-word sequence
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (1, 50, 20) — hidden state at each step
print(c_n.shape)     # (1, 1, 20) — cell state: the long-range memory highway
```

LSTMs became dominant from ~2014 onward. Google Translate, Siri, and most voice assistants used LSTMs until the Transformer arrived.

### GRU — The Lighter Alternative *(2014)*

Cho et al. noticed that LSTMs have four sets of weight matrices per layer — computationally heavy. They proposed the **Gated Recurrent Unit (GRU)**, which merges the forget and input gates into a single **update gate** and removes the separate cell state.

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(1, 50, 10)
output, h_n = gru(x)
print(output.shape)  # (1, 50, 20)
```

Two gates instead of three. No separate cell state. Trains ~30% faster with similar quality on most tasks. The choice between LSTM and GRU is empirical — try both and measure.

---

## Word Embeddings: Teaching Machines That Words Have Meaning

One-hot encoding represents each word as a vector with a 1 in one position and zeros everywhere else. In a vocabulary of 10,000 words, "cat" and "kitten" are represented as completely orthogonal vectors — their dot product is exactly zero. The model has been told, mathematically, that these two words have *nothing in common*.

That is obviously wrong. And the fix changed everything.

### Word2Vec: "You Shall Know a Word by the Company It Keeps" *(2013, Mikolov et al.)*

The idea behind Word2Vec came from linguistics: words that appear in similar contexts tend to have similar meanings. "I adopted a **cat**" and "I adopted a **kitten**" and "I adopted a **puppy**" — because "cat", "kitten", and "puppy" all appear near "adopted", they must be semantically related. Word2Vec operationalises this intuition.

Instead of hand-designing what words mean, Word2Vec trains a tiny neural network on a prediction task:

- **Skip-gram:** Given the word "cat", predict the surrounding words: "adopted", "fluffy", "sleeping"
- **CBOW (Continuous Bag of Words):** Given the surrounding words, predict "cat"

The model never actually uses the predictions — it's a trick. What matters are the weights learned by the hidden layer. Those weights become the word embeddings: dense 100–300 dimensional vectors where semantically similar words cluster together.

```
   Skip-gram:   [target word]  ─────────────────────►  [predict context words]
   CBOW:        [context words] ──► [average/sum] ────►  [predict target word]
```

After training, the famous discovery: the vector arithmetic captures real-world relationships.

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

The embedding space has learned gender as a direction, royalty as another direction, and country-capital pairs as yet another. Nobody told it these relationships exist — they emerged from predicting surrounding words in a large corpus.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "land"],
    ["man", "and", "woman", "are", "humans"],
    ["the", "prince", "is", "the", "son", "of", "the", "king"],
]

model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, epochs=200)

print("king ↔ queen similarity:", model.wv.similarity("king", "queen"))
print("Words most similar to 'king':", model.wv.most_similar("king", topn=3))
```

### GloVe: Global Co-occurrence *(2014, Pennington et al.)*

Word2Vec only looks at local windows of a few words. **GloVe** uses the *entire corpus* at once. It builds a global word-word co-occurrence matrix $X$ where $X_{ij}$ = how many times word $j$ appeared near word $i$ across the full corpus, then factorises it to learn embeddings that capture both local and global statistics.

The practical difference: GloVe embeddings often have better geometric structure for downstream tasks; Word2Vec trains faster on streaming data. Both were superseded by contextual embeddings (BERT, Module 5), but understanding them is essential for understanding what embeddings *represent*.

### FastText: Words Are Made of Parts *(2016, Bojanowski et al.)*

Word2Vec assigns one fixed vector per word. "run" and "running" are completely independent. Words never seen in training ("transformerless", "GPT-4ification") get no vector at all — the **out-of-vocabulary (OOV)** problem.

**FastText** (Facebook AI Research) solves this by representing every word as the *sum of its character n-gram embeddings*. For "where" with $n=3$:

$$\langle\text{wh}, \text{whe}, \text{her}, \text{ere}, \text{re}\rangle + \langle\text{where}\rangle$$

For any new word — even a made-up one — you can construct its vector from familiar subword pieces. This makes FastText robust to typos, morphological variants, and domain-specific jargon. It became the default embedding choice for many production NLP systems before BERT arrived.

---

## Seq2Seq: One Sequence In, One Sequence Out *(2014)*

Machine translation needs to convert a sentence of one length in English to a sentence of potentially different length in French. An RNN predicts one output per input position — wrong shape. You need a model that accepts a variable-length input and produces a variable-length output.

The solution: two separate RNNs working together.

**The Encoder** reads the entire input sentence word by word, building up a hidden state that ideally captures the full meaning. At the end, this final hidden state — called the **context vector** — is a compressed summary of the entire input.

**The Decoder** is given that context vector as its initial state and generates output words one at a time, each word feeding back as input for the next step, until it generates a special end-of-sentence token.

```
"The cat sat on the mat"
         │
    [Encoder LSTM]
         │  (reads word by word, builds hidden state)
         ▼
  [context vector]  ← compressed meaning of entire input
         │
    [Decoder LSTM]
         │  (generates one word at a time)
         ▼
"Le chat était assis sur le tapis"
```

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # returns all hidden states + final (h_n, c_n)
        return self.lstm(self.embed(x))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)  # predict next word

    def forward(self, x, hidden):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden
```

Seq2Seq in 2014 dramatically improved machine translation quality. But it had a critical flaw: the entire meaning of the input sentence had to fit into one fixed-size context vector. For a 3-word sentence, fine. For a 50-word sentence with complex structure, crucial information got crushed out.

The decoder — trying to translate word 47 — only had access to that single compressed summary, not to the original encoder state at word 3 where the relevant subject was introduced. Quality collapsed on long inputs.

---

## Bahdanau Attention: The Bridge to Transformers *(2014)*

Dzmitry Bahdanau's paper "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) is one of the most important NLP papers ever written — not because it invented attention (the idea existed earlier in other forms), but because it demonstrated definitively that **a decoder can look directly at any part of the input at each output step**, rather than being constrained to a single bottleneck vector.

The intuition is natural: when a human translator translates the 5th word of an English sentence into French, they don't just remember a compressed summary of the English — they *look back* at the original English, focusing their attention on the specific words most relevant to the current output word.

### How Bahdanau Attention Works

The encoder still runs through the input, but now we keep *all* encoder hidden states, not just the last one. The encoder produces a sequence of hidden states $[h_1, h_2, \ldots, h_n]$, one per input token.

At each decoder step $t$, before generating the next output word, the decoder:

1. **Scores** every encoder hidden state $h_i$ against the current decoder state $s_{t-1}$: how relevant is input word $i$ for what I'm about to generate?
2. **Normalizes** scores via softmax → attention weights $\alpha_{t,i}$ (sum to 1)
3. **Computes** a **context vector** $c_t$ = weighted sum of encoder states
4. **Concatenates** $c_t$ with the decoder state to predict the next output word

$$e_{t,i} = \text{score}(s_{t-1}, h_i) \quad \text{(alignment score)}$$
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})} \quad \text{(softmax → attention weights)}$$
$$c_t = \sum_i \alpha_{t,i} h_i \quad \text{(weighted context vector)}$$

```python
import torch
import torch.nn.functional as F

def bahdanau_attention(encoder_outputs, decoder_hidden):
    # encoder_outputs: all encoder states [batch, src_len, hidden]
    # decoder_hidden:  current decoder state [batch, 1, hidden]

    # Score each encoder position against the current decoder state
    scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
    # scores: [batch, 1, src_len]

    # Softmax: which input words to focus on right now?
    weights = F.softmax(scores, dim=-1)

    # Weighted sum: blend encoder states by attention weights
    context = torch.bmm(weights, encoder_outputs)
    # context: [batch, 1, hidden] — dynamic, focused summary for this step

    return context, weights
```

The attention weights are directly interpretable. You can visualise a matrix where rows = output words and columns = input words, and each cell shows how much the decoder focused on that input word when generating that output word. For English → French translation, you can literally see the model aligning "The cat" → "Le chat", "sat" → "était assis."

This interpretability was a revelation. For the first time, researchers could see *what* the model was paying attention to — not just what it output.

### The Question That Broke Everything Open

Bahdanau attention made Seq2Seq dramatically better. Translation quality on long sentences improved sharply. The paper was widely reproduced and extended (Luong et al. 2015 introduced a simpler variant with dot-product scoring).

But researchers staring at these results noticed something uncomfortable: the LSTM layers were still there, still processing sequentially, still preventing parallelisation. And the attention mechanism was doing all the interesting relational work — connecting input to output across arbitrary distances.

Then someone asked the question that changed everything:

*"If attention is doing all the heavy lifting — connecting distant tokens, managing long-range dependencies — why do we still need the sequential LSTM at all?"*

That question was the seed of *"Attention Is All You Need"* (2017). Module 5 is the answer.

---

## NLP Evaluation Metrics

Classification NLP tasks (sentiment, intent detection) use F1 and accuracy from Module 1. But generative NLP tasks — translation, summarisation — produce free-form text. You cannot score "Le chat était assis sur le tapis" as simply right or wrong. These metrics were designed for that problem.

### BLEU — Bilingual Evaluation Understudy *(2002)*

Your translation model outputs: *"The cat is on the mat."*  
Human reference: *"The cat sat on the mat."*

BLEU measures n-gram overlap between generated text and one or more human references:

```
Reference:  "The cat sat on the mat"
Generated:  "The cat is on the mat"

1-gram overlap: The, cat, on, the, mat → 5/6 = 0.83
2-gram overlap: "The cat", "on the", "the mat" → 3/5 = 0.60
BLEU ≈ geometric mean of 1-gram through 4-gram precisions × brevity penalty
```

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate  =  ["the", "cat", "is",  "on", "the", "mat"]

score = sentence_bleu(reference, candidate)
print(f"BLEU: {score:.3f}")  # ~0.63
```

**Score interpretation:** 0–100. Above 30 is considered decent machine translation. State-of-the-art reaches 40–50 on standard benchmarks.

**Failure mode:** BLEU only measures surface overlap, not meaning. "The feline reclined upon the carpet" and "The cat sat on the mat" express the same thing but score very low. BLEU also rewards short outputs (adding a brevity penalty helps but doesn't fully fix this). It has been supplemented in modern research by semantic metrics like **BERTScore** (which uses contextual embeddings to measure meaning similarity, not word overlap).

### ROUGE — Recall-Oriented Understudy *(2004)*

Where BLEU emphasises *precision* (did the generated text use the right words?), ROUGE emphasises *recall* (did the generated text cover the important words from the reference?). Recall is more natural for **summarisation** — you care whether the summary captures key information, not whether it invented extra words.

- **ROUGE-1:** Unigram overlap (individual words)
- **ROUGE-2:** Bigram overlap (two-word phrases)
- **ROUGE-L:** Longest Common Subsequence — captures structure even without exact phrase matches

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

reference = "The quick brown fox jumps over the lazy dog near the river"
generated = "The fox jumps over the lazy dog"

scores = scorer.score(reference, generated)
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")  # word-level coverage
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")  # structural similarity
```

**Failure mode:** Like BLEU, ROUGE measures surface overlap. A high ROUGE score doesn't guarantee a *useful* summary — a model could score well by copying sentences verbatim from the source, even if those sentences aren't the most important ones.

### When to Use Which NLP Metric

| Task | Primary Metric | Why |
|------|----------------|-----|
| Machine translation | BLEU | Standard benchmark across papers |
| Summarisation | ROUGE-L | Coverage of key content |
| Question answering | Exact Match + F1 | Whether the answer span overlaps reference |
| Semantic similarity | BERTScore | Captures meaning, not just surface overlap |
| LLM output quality | Human eval + LLM-as-judge | No automatic metric captures helpfulness fully |

> **Perplexity** — how surprised the language model is by test text — is the core *intrinsic* metric for language models and is covered in Module 6.

---

## Quick Reference — Module 4 Q&A

**Q: Why can't you use a CNN for text?**  
**A:** CNNs need fixed-size inputs — text has variable length. More critically, CNNs capture local spatial patterns (adjacent pixels). Text meaning often depends on words that are 10, 20, or 50 positions apart. A CNN's local receptive field misses these long-range dependencies unless you stack many layers, which becomes slow and unstable.

**Q: What problem does TF-IDF solve that BoW doesn't?**  
**A:** BoW weights "the" and "photosynthesis" equally if they appear the same number of times. TF-IDF down-weights words that appear in almost every document (low discriminative power) and up-weights words that appear frequently in this document but rarely elsewhere. It highlights what's distinctive about a document.

**Q: Why do RNNs have a long-range problem?**  
**A:** The hidden state is a fixed-size vector that gets partially overwritten at every step. After 50 updates, information from step 1 has been diluted to near-zero. Backpropagating through 50 steps also means multiplying 50 weight matrices — gradients vanish exponentially. The model literally cannot learn correlations between tokens that are far apart.

**Q: What do the three LSTM gates actually do?**  
**A:** The **forget gate** decides what to erase from the long-term cell state (past context that's no longer relevant). The **input gate** decides what new information to write into the cell state. The **output gate** decides what part of the cell state to expose as the hidden state for this step. Together, they give the LSTM surgical control over what to remember and for how long.

**Q: How does "king − man + woman ≈ queen" work?**  
**A:** Word2Vec learns embeddings by predicting surrounding words. "King" and "queen" appear in similar contexts (throne, crown, reign). "Man" and "woman" appear in similar contexts (person, human, adult). The embedding space learns "royalty" as one direction and "gender" as another. Subtracting the "man" direction and adding the "woman" direction navigates from the male-royalty region to the female-royalty region — which happens to be where "queen" lives.

**Q: What is Bahdanau attention and why does it matter?**  
**A:** Bahdanau attention allows the decoder to look at all encoder hidden states at each generation step, computing a weighted sum based on relevance to the current output. It eliminates the fixed-size bottleneck of Seq2Seq. Its deeper importance: it demonstrated that attention alone can capture long-range linguistic dependencies better than recurrence — which directly led researchers to ask whether the RNN was even necessary, and ultimately to the Transformer.

---

## Bridge to Module 5

By the end of this module, you have a sequence of engineering solutions to the same core problem:
- BoW: no memory, no order
- RNN: memory, but it fades over distance
- LSTM/GRU: better memory with gates, but still sequential and slow
- Attention bolted onto LSTM: direct token-to-token connections, but recurrence still forces sequential processing

The trajectory is clear: each generation kept the useful part (memory, long-range connections) and threw away the limiting part. The final step — in 2017 — was throwing away the recurrence itself. What remained was attention, stacked many times, processing all tokens in parallel.

That is Module 5.