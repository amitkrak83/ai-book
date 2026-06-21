# Module 6 — Large Language Models *(2018 – 2022)*

**Why this era exists:** The Transformer proved itself on translation. The next question: what happens if you scale up a decoder-only Transformer enormously and train it to predict the next word on most of the internet? The answer turned out to be: general-purpose language capability at a level nobody had seen before.

---

# Module 6 — Part 1: Core LLM Concepts

> **What this part covers:** How LLMs are trained, what makes them so capable, how to work with them (fine-tuning, prompting, RAG), and where they break down. This is the foundation — everything in Part 2 builds on this.

---

## Pretraining *(GPT-1 2018, BERT 2018)*

### Why We Stopped Labeling Everything
Until 2018, natural language processing was stuck in a labeling bottleneck. If you wanted to classify customer feedback, you had to hire humans to tag 50,000 sentences. If you wanted translation, you had to curate millions of parallel sentence pairs. This was incredibly slow, expensive, and didn't scale. 

Then came a simple, revolutionary realization: **what if the text itself was the label?** 

Every sentence on the internet contains hidden training data. If you write *"The sky is [blank]"*, the correct continuation is naturally *"blue"*. By training a model on the single task of **predicting the next word**, it is forced to learn grammar, geography, history, and reasoning as a side-effect. This is called **self-supervised pretraining**, and it unlocked the ability to train on the entire public internet without paying for human labels.

### Autocomplete on Steroids
Imagine a super-powered autocomplete engine. Pretraining is the phase where this engine reads billions of pages of books, Wikipedia, news articles, and code repositories. 

The model starts completely random, guessing gibberish. Every time it gets a word wrong, it adjusts its internal weights using gradient descent. Over months of reading, it builds a deep statistical map of how human language works.

### How Pretraining Works
The two primary styles of pretraining that dominated this era are:

1.  **Causal Language Modeling (GPT-style):** The model reads left-to-right, looking at past words to predict the next single word. It is restricted from seeing the future.
2.  **Masked Language Modeling (BERT-style):** The model reads bidirectionally, looking at both left and right contexts to fill in a blank space (masked word) in the middle of a sentence.

```python
# The pretraining objective is simple — just next token prediction
# loss = CrossEntropy(model(tokens[:-1]), tokens[1:])
import torch
import torch.nn.functional as F

# A simple training step for a next-word predictor
tokens = [101, 2003, 1037, 2733, 4005, 2058] # "The quick brown fox jumps over" in integer IDs
input_ids  = torch.tensor(tokens[:-1]).unsqueeze(0) # Inputs: "The quick brown fox jumps"
target_ids = torch.tensor(tokens[1:]).unsqueeze(0)  # Targets: "quick brown fox jumps over"

# Let's assume a vocabulary size of 30,522 tokens
vocab_size = 30522
# The model outputs a probability score (logits) for every word in the vocabulary at each position
logits = torch.randn(1, 5, vocab_size) # [batch, sequence_length, vocab_size]

# Calculate cross-entropy loss between what the model guessed and the actual next words
loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
print("Pretraining loss:", loss.item())
```

---

## Parameters — The Dials of the Brain

### The Soundboard Metaphor
When engineers say a model has "175 billion parameters," think of it like a **colossal sound mixing console** with 175 billion individual dials, sliders, and knobs. 

```
  [Input Text] ──► ( Dial 1 ) ──► ( Dial 2 ) ──► ... ──► [Next Word]
                      ▲              ▲
                      │              │
                (Adjusted by training gradients)
```

Each parameter is a single decimal number (a weight or a bias). When text enters the model, it is multiplied and added by these parameter values in billions of combinations. If a dial is turned slightly to the left, the word "blue" becomes a tiny bit more likely. If turned to the right, "green" is preferred. 

Pretraining is the process of fine-tuning all 175 billion dials until the console output matches human writing.

### How the Dials Scaled
As the number of dials (parameters) grew, the model's capacity to store complex concepts and patterns exploded:

| Model | Year | Parameter Count | Analogy |
| :--- | :--- | :--- | :--- |
| **GPT-1** | 2018 | 117 Million | A small local radio mixing board |
| **BERT-Large** | 2018 | 340 Million | A recording studio mixing console |
| **GPT-2** | 2019 | 1.5 Billion | A major TV broadcast control room |
| **GPT-3** | 2020 | 175 Billion | An entire space shuttle launch control center |
| **GPT-4 (est.)** | 2023 | ~1 Trillion | A global network of control rooms |

*What breaks at scale?* More parameters mean the model can memorize and generate higher-quality text, but the hardware requirements skyrocket. A 175-billion-parameter model cannot fit in the memory of a single standard computer; it must be split across dozens of high-end enterprise GPUs, introducing massive infrastructure complexity.

---

## Scaling Laws — The Baking Ratio

### The predictable recipe
Before 2020, scaling up models was mostly trial-and-error. Researchers didn't know if adding more parameters would hit a sudden wall of diminishing returns. In 2020, Jared Kaplan and a team at OpenAI published a paper showing that LLM performance isn't random—it follows a predictable power law.

Think of it like **baking a cake**: you need the perfect ratio of flour (model parameters $N$), sugar (training data size in tokens $D$), and oven time (total compute budget $C$). If you increase the size of the cake pan but don't add more sugar or flour, the cake will collapse.

```
                  [Compute Budget (C)]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     [Model Size (N)]  ◄────────►  [Dataset Size (D)]
          Parameters                    Tokens
```

The mathematical scaling law proved that:

$$\text{Loss} \propto N^{-\alpha_N}, \quad \text{Loss} \propto D^{-\alpha_D}, \quad \text{Loss} \propto C^{-\alpha_C}$$

As long as you scale parameters ($N$), dataset tokens ($D$), and compute ($C$) in harmony, the loss (error rate) decreases along a predictable straight line on a logarithmic graph.

### The Chinchilla Correction (2022)
In 2022, DeepMind published the "Chinchilla" paper, which corrected a major flaw in how scaling laws were being applied. They showed that models like GPT-3 were **undertrained**: companies were building massive models ($N$) but not feeding them enough data ($D$) because data was hard to clean. 

The Chinchilla scaling law proved that for optimal performance, **parameters and tokens should scale in a 1:1 ratio**. If you double your compute budget, you should make the model 1.4x larger and train it on 1.4x more data. This shift led to smaller, denser models (like LLaMA) that were trained for much longer on massive datasets, making them cheaper to run in production.

---

## Tokenization — Words as Lego Blocks

### Why We Can't Use Letters or Whole Words
Computers do not understand letters or words directly. We must convert text into numbers.
*   **Attempt 1: Character-level (A, B, C...)**: The vocabulary is tiny (26 letters + punctuation), but sequences become incredibly long. A single sentence might be 300 characters long, slowing down self-attention quadratically.
*   **Attempt 2: Word-level (Apple, Banana...)**: The vocabulary becomes massive (millions of words). The model cannot handle typos, and new slang words (like *"rizz"*) require changing the whole vocabulary dictionary.

The solution is **subword tokenization**: we chop words into common building blocks, like **Lego bricks**. The word *"tokenization"* gets split into `["token", "ization"]`. The word *"unbelievable"* becomes `["un", "believ", "able"]`. This allows a vocabulary of just 50,000 tokens to represent every possible English word, typo, and code syntax.

```
  Raw Text:     "Unhappy coding!"
                   │
  Tokenized:    ["Un", "happy", " coding", "!"]  (Lego pieces)
                   │
  Integer IDs:  [3421,  12093,    20392,  0]    (Computer numbers)
```

### Byte-Pair Encoding (BPE)
BPE is the most common algorithm for building these subword vocabularies. It starts with single characters and iteratively merges the most frequent pairs of characters it sees in its training text until it reaches its target vocabulary size.

```python
from transformers import GPT2Tokenizer

# Let's inspect how a real tokenizer slices text
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Tokenization splits text into subword tokens."
token_ids = tokenizer.encode(text)

print("Integer IDs representing the text:\n", token_ids)
print("\nSubword chunks (notice the 'Ġ' representing space boundaries):\n", 
      [tokenizer.convert_ids_to_tokens(i) for i in token_ids])
```

*Failure Mode:* Because tokenizers see subwords rather than characters, LLMs are notoriously bad at character-level tasks. Ask an LLM *"How many 'r's are in 'strawberry'?"* and it will often say two. Why? Because it sees the word `strawberry` as two blocks: `["straw", "berry"]`. It has to guess how many characters are inside those blocks.

---

## Context Window — The Desk Space

### The Metaphor of the Work Desk
Imagine you are a researcher writing a report. Your **context window** is the size of your **work desk**. 

*   If you have a tiny desk, you can only keep one page open at a time. If you want to write a summary of a book, you have to read a page, memorize the key points, throw it away, and open the next. You will inevitably forget details.
*   If you have a massive desk, you can lay out the entire book side-by-side, comparing Chapter 1 directly to Chapter 10.

```
       Context Window (Desk Space)
  ┌─────────────────────────────────────┐
  │ [Prompt]  [Working History]  [Task] │ ◄── Fits inside Context Window
  └─────────────────────────────────────┘
  =======================================
  [Old conversation tokens...]           ◄── Falls off the desk (forgotten)
```

In a Transformer, the context window is the maximum sequence length $N$ that the model can process in a single forward pass. Because attention memory grows quadratically ($O(N^2)$), expanding the desk space requires massive amounts of GPU memory.

### The Limits of Memory
If a model has a context window of 2,048 tokens, you cannot paste a 10,000-word PDF into it—it will physically fail to compute. If you continue a chat conversation past the context limit, the earliest messages "fall off the desk." The model completely forgets who you are or what you asked five minutes ago.

---

## Embeddings — The Semantic GPS

### Moving Beyond Word2Vec
In Module 4, we saw that Word2Vec gives every word a fixed vector coordinate. However, language is context-dependent. Consider the word **"bank"**:
1.  *"I sat on the grassy **bank** of the river."*
2.  *"I deposited money at the local **bank**."*

In Word2Vec, both "banks" have the exact same coordinates. In a modern LLM, **contextual embeddings** dynamically shift coordinates depending on surrounding tokens. 

### Coordinates in Semantic Space
Think of embeddings as **GPS coordinates in a high-dimensional concept space**. In this space, the dimension isn't latitude and longitude, but concepts like "financial-ness," "nature-ness," or "activity." The word "bank" in sentence 1 is placed near "river" and "forest"; in sentence 2, it is pulled toward "vault" and "finance."

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load a small, efficient embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Average the token embeddings to get a single sentence vector
    return outputs.last_hidden_state.mean(dim=1)

# Embed three sentences
e1 = get_embedding("The banker counted the money in the vault.")
e2 = get_embedding("The river bank was muddy and green.")
e3 = get_embedding("The financial institution held his deposits.")

# Calculate similarity (angle between coordinates)
finance_similarity = torch.cosine_similarity(e1, e3).item()
nature_similarity = torch.cosine_similarity(e1, e2).item()

print(f"Bank(vault) vs Institution (Finance): {finance_similarity:.3f}") # High similarity
print(f"Bank(vault) vs Bank(river) (Nature):    {nature_similarity:.3f}")  # Low similarity
```

---

## Fine-Tuning — Behavioral Training

### The Law Student Analogy
What is the difference between Pretraining and Fine-Tuning?
*   **Pretraining (General Education):** A student spends years reading every book, article, and essay in the university library. They learn vocabulary, grammar, science, and history. They are highly intelligent, but if you ask them to write a legal contract, they will write a rambling story about contracts because they don't know the professional format.
*   **Fine-Tuning (Law School):** You take this well-read student and put them through a focused, intense law school program. You don't teach them how to read or write; you teach them the *structure, tone, and behavior* expected of a lawyer.

```
  [Pretrained Base Model] ──► (Train on Labeled Task Data) ──► [Fine-tuned Specialist]
     Knows language rules        Updates specific weights         Knows how to format
     and general facts           to specialize behavior          or act in a target domain
```

### Full Fine-Tuning vs. PEFT (LoRA)
*   **Full Fine-Tuning:** You update all 175 billion parameters of the model. This is like forcing the student to rewrite their entire brain for law school. It is incredibly memory-intensive and expensive.
*   **Parameter-Efficient Fine-Tuning (LoRA):** You freeze all the original weights and attach a small "adapter" (like a sticky-note pad) next to the key layers. You only train the adapter. This keeps training cheap and fast while maintaining the base model's general knowledge.

```python
# LoRA concept — manual implementation
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4):
        super().__init__()
        d_in  = original_linear.in_features
        d_out = original_linear.out_features
        
        self.original = original_linear  # Freeze this original linear layer
        for p in self.original.parameters():
            p.requires_grad = False
            
        # Create two small low-rank matrices: A and B
        # Input maps from d_in -> rank (small), then from rank -> d_out
        self.lora_A   = nn.Linear(d_in, rank, bias=False)
        self.lora_B   = nn.Linear(rank, d_out, bias=False)
        
        # Initialize B to zero so that the initial adapter output is exactly 0
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # The output is the original layer prediction + the tiny adapter correction
        return self.original(x) + self.lora_B(self.lora_A(x))
```

---

## RLHF — The Etiquette Coach

### Autocomplete's Dark Side
During pretraining, a model reads the raw internet. This means it learns exactly how humans write online—including conspiracy theories, toxic forum arguments, and malware instructions. 

If you ask a raw pretrained model: *"How do I hotwire a car?"*, it might complete the text with a step-by-step tutorial because that is what statistically follows the prompt on the internet. 

To make models safe and useful, we need a way to steer their behavior. This process is called **Alignment**, and the dominant technique is **RLHF (Reinforcement Learning from Human Feedback)**.

```
  1. Base Model ──► 2. Supervised Fine-Tuning (SFT) ──► 3. Reward Modeling ──► 4. PPO RL
     Autocomplete    Answers like a helpful assistant     Human: "A is better   Adjust dials to
     internet text   on curated prompt-answers            than B" -> Train RM   maximize score
```

### The Three Steps of RLHF
1.  **Supervised Fine-Tuning (SFT):** We hire humans to write high-quality prompts and matching answers (e.g., *"Prompt: Write a poem about rain. Answer: [Beautiful, safe poem]"*). We fine-tune the model on this dataset.
2.  **Reward Model Training:** We prompt the model to generate multiple answers, and ask human labelers to rank them from best to worst. We use this ranking to train a separate "Reward Model" to predict how much a human would like any given text.
3.  **Reinforcement Learning (PPO):** We let the LLM generate text. The Reward Model evaluates it and gives a score (+1 for helpful/safe, -1 for toxic). The LLM uses this score to adjust its weights using reinforcement learning, learning to seek rewards.

---

## Hallucination — The Confident Improviser

### Why LLMs Lie
Hallucination is the phenomenon where an LLM confidently states facts that are completely made up. 

It is crucial to understand that **hallucination is not a bug—it is the core mechanism of how LLMs work**. An LLM does not have a database of facts or a connection to reality inside its weights. It only has statistical associations between words. 

Think of the model as a **confident improvisational actor** on stage. If you ask it: *"Who was the President of the United States in 1897?"*, it will predict the most statistically plausible sequence of words that follows. If its training data contains strong associations, it will say *"William McKinley"*. If it is uncertain, it will still generate a name that *looks* like a president, like *"John Harrison"*, with absolute confidence. It cannot check its own work because there is no internal "fact-checker" layer.

### Mitigation Strategies
*   **Grounding (RAG):** Give the model a textbook containing the answer and ask it to read from it.
*   **Formatting Constraints:** Force the model to output steps, cite its sources, or output in JSON.
*   **Verification loops:** Ask the model to double-check its own logic in a second pass.

---

## Context Bias — Lost in the Middle

### The "Attention Span" Curve
Even if you have a massive context window (desk space), the model does not pay equal attention to everything on the desk. 

In 2023, researchers discovered the **"Lost in the Middle"** effect: LLMs are highly skilled at retrieving facts at the very beginning of the prompt (the start of the conversation) or the very end (the instruction they just read). However, information buried in the middle of a long context window is often completely ignored.

```
  High Attention ──┐                 ┌── High Attention
                   │                 │
                   ▼                 ▼
             [Start of prompt] ... [Middle of prompt] ... [End of prompt]
                                         ▲
                                         │
                                   Lost in the Middle
                                     (Low Attention)
```

*Why does this happen?* During pretraining, models are trained on documents (like articles and books) where the most important information is usually in the introduction (start) or the conclusion (end). The model learns an inductive bias to pay less attention to the middle. When building applications, engineers must place critical instructions and facts at the very beginning or end of the prompt to avoid this bias.

---

## Retrieval-Augmented Generation (RAG) — The Open Book Exam

### Why We Need RAG
LLMs suffer from three massive real-world problems:
1.  **Cutoff dates:** Once training is done, the model's weights are locked. It knows nothing about events that happened yesterday.
2.  **Private data:** The model was trained on the public internet. It knows nothing about your company's internal reports or private customer data.
3.  **Hallucinations:** When it doesn't know an answer, it invents one.

**RAG (Retrieval-Augmented Generation)** resolves this by turning the model's task from a **closed-book exam** (relying on memory) into an **open-book exam** (giving it the reference document to read).

```
  [User Query] ──► (1. Search System) ──► [Find Relevant Page]
                                                │
                                                ▼
  [Answer] ◄── (3. LLM Generates) ◄── (2. Paste into Prompt)
```

### The RAG Pipeline Steps

1.  **Chunking:** We take a massive document (e.g., a 100-page manual) and chop it into small, readable paragraphs (chunks) of around 500 characters.
2.  **Vector Store Indexing:** We convert each chunk into an embedding vector (GPS coordinates) and store them in a database designed for vector lookups.
3.  **Retrieval:** When a user asks: *"What is the policy on maternity leave?"*, we embed the query and find the top 3 chunks in our database that have the closest coordinates.
4.  **Generation:** We paste those 3 chunks into the LLM prompt:
    *"Here is the context: [Insert maternity chunks]. Question: What is the policy? Answer using the context only."*
5.  **Output:** The LLM reads the context and writes a factually grounded answer.

```python
# A conceptual complete RAG retrieval & generation loop
import numpy as np

# Let's mock a simple database of document chunks
document_chunks = [
    "Maternity leave policy: Employees get 12 weeks of fully paid leave.",
    "Expense policy: Travel meals are covered up to $50 per day.",
    "WFH policy: Employees can work remotely up to 2 days per week."
]

# Mock embeddings (simple 3D vectors representing topics: [parental, finance, workspace])
database_embeddings = np.array([
    [0.9, 0.1, 0.1],  # Chunks 0: high parental
    [0.1, 0.9, 0.1],  # Chunk 1: high finance
    [0.1, 0.2, 0.8]   # Chunk 2: high workspace
])

def basic_retrieve(query_vector, k=1):
    # Compute similarity between query and all chunks using dot product
    similarities = np.dot(database_embeddings, query_vector)
    best_match_idx = np.argmax(similarities)
    return document_chunks[best_match_idx]

# User asks about parental benefits
query_vector = np.array([0.95, 0.05, 0.0]) # Emphasizes parental topic
retrieved_context = basic_retrieve(query_vector)

# Now, we would feed this context directly to the LLM
prompt = f"Context: {retrieved_context}\nQuestion: How long is maternity leave?\nAnswer:"
print("LLM Prompt:\n", prompt)
```

---

## Key Model Milestones

| Model | Year | Key Contribution | Analogy |
| :--- | :--- | :--- | :--- |
| **GPT-1** | 2018 | Proved pretraining + fine-tuning works. | The first prototype engine. |
| **BERT** | 2018 | Dominated search and classification using bidirectional mask training. | The detective who reads both sides. |
| **GPT-2** | 2019 | Proved scaling parameters generates high-quality, coherent essays. | The creative writer who surprised everyone. |
| **GPT-3** | 2020 | Showed that scale unlocks "In-Context Learning" (zero-shot capabilities). | The giant calculator that does everything. |
| **InstructGPT** | 2022 | Applied RLHF to align next-token predictions with helpful human goals. | Autocomplete gets a helpful tutor. |

---

---

# Module 6 — Part 2: Advanced LLM Techniques

> **What this part covers:** Once you understand how LLMs work, the next questions are: How do you make them smaller and faster? How do you do RAG at production scale? How do you use models that reason through hard problems? This part covers the engineering and research techniques that separate prototype from production.

---

# LLM Efficiency — Compressing the Giants

### The Hardware Wall
Let's make the memory problem concrete:
*   **LLaMA-2 70B** has 70 billion parameters.
*   In standard training (32-bit float), each parameter takes **4 bytes** of memory.
*   To load the model into VRAM, you need: $70\text{B} \times 4\text{ bytes} = 280\text{ GB}$ of VRAM.
*   An enterprise A100 GPU has 80 GB of VRAM. You would need **4 A100s** just to load the model for inference.
*   Fine-tuning requires storing gradients and optimizer states, which multiplies VRAM needs by 4x. You'd need **16 A100 GPUs** (costing ~$160,000 to purchase or ~$128/hr to rent) just to train.

To democratize AI and run models on standard hardware, engineers developed the efficiency techniques detailed below.

---

## Fine-Tuning Efficiently: LoRA & QLoRA

### LoRA (Low-Rank Adaptation)
As shown in Part 1, LoRA freezes the original large matrix $W$ and trains a tiny adapter matrix $B \times A$. By setting the rank $r$ to a small number (e.g., 8), we reduce the number of trainable weights by over 99%. 
*   **Inference merge:** When training is done, we can add $W_{\text{new}} = W_{\text{frozen}} + (B \times A)$ once. During deployment, the model runs at the exact same speed as the original model with zero latency overhead.

### QLoRA (Quantized LoRA)
QLoRA takes this a step further: it loads the base model in **4-bit quantized format** (shrinking a 28 GB model to 4 GB) but runs the LoRA adapter updates in **16-bit precision**. 
*   This breakthrough makes it possible to fine-tune a 7-billion-parameter model on a single consumer gaming GPU (like an RTX 3090 or 4090) instead of a $10,000 enterprise card.

```python
# pip install bitsandbytes peft transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Step 1: Load base model in compressed 4-bit NormalFloat format
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Quantize quantization constants for extra VRAM savings
    bnb_4bit_quant_type="nf4",       # NormalFloat4: optimal format for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 2: Prepare model layers for low-precision gradient updates
model = prepare_model_for_kbit_training(model)

# Step 3: Attach trainable LoRA adapters to attention projections
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,540,389,888 || trainable%: 0.12%
```

---

## Quantization — Rounding the Decimals

### The Checkout Analogy
Imagine buying groceries. The items cost $1.99, $3.02, $5.97, and $0.98. If you sum them exactly, it's $11.96. If you round each to the nearest dollar ($2, $3, $6, $1) and sum them, it is $12.00. The rounded calculation is 10x faster to do in your head, and the error (4 cents) is negligible for most decisions.

**Quantization** is the mathematical equivalent of rounding weights from high-precision floating points (FP32) to smaller formats like 8-bit integers (INT8) or 4-bit formats (INT4).

```
  Precision Format   Size per Weight   7B Model VRAM Needs
  ────────────────────────────────────────────────────────
  32-bit (FP32)       4 bytes           28 GB (Full Precision)
  16-bit (FP16)       2 bytes           14 GB (Standard serving)
  8-bit (INT8)        1 byte            7 GB  (Zero quality loss)
  4-bit (INT4)        0.5 bytes         3.5 GB (Fits on a laptop/phone)
```

### Quantization Styles
1.  **Post-Training Quantization (PTQ):** Take a fully trained 16-bit model and round its weights directly to 4-bit/8-bit. This takes minutes but can cause slight accuracy drops on complex reasoning tasks.
2.  **Quantization-Aware Training (QAT):** Simulate the rounding errors during the training process. The model learns to adjust its other weights to compensate for the lost precision, preserving near-perfect accuracy.

---

## Knowledge Distillation — Professor and Apprentice

### The Teacher-Student Framework
Large models (like GPT-4) are excellent teachers because they don't just output correct answers—they contain rich probability distributions over what makes an answer correct. **Knowledge Distillation** trains a small "student" model to mimic a large "teacher" model's outputs.

```
  [Cat Photo] ──► [Teacher (GPT-4)] ──► [Soft Targets] ──► [Student Model]
                                          Cat:  88%           Learn soft
                                          Dog:  11%           probabilities
                                          Car:   1%           instead of raw labels
```

If we train the student model on raw hard labels (e.g., `"Classification: Cat"`), it learns nothing about the relationships between categories. 
If we train the student model on the teacher's **soft targets** (e.g., `"Cat: 88%, Dog: 11%, Car: 1%"`), the student learns that a cat looks somewhat like a dog, but nothing like a car. This "dark knowledge" allows the student to achieve near-teacher performance with a fraction of the size.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.7):
    """
    Computes a combined loss to train a student model based on a teacher's knowledge.
    temperature: Softens the teacher's probability distribution to reveal subtle details.
    alpha: Controls the balance between mimicking the teacher and predicting the raw ground-truth.
    """
    # Step 1: Compute softened teacher probabilities
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    # Step 2: Compute softened student log-probabilities
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    
    # Step 3: Compute Kullback-Leibler divergence (similarity between probability distributions)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # Step 4: Compute standard cross-entropy loss against ground truth labels
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    # Return weighted sum of both losses
    return alpha * distill_loss + (1 - alpha) * hard_loss
```

---

## Pruning — Trimming the Overgrown Brain

### Cutting the Dead Wood
During training, many weights in a model become extremely small, close to zero. They sit in memory, consuming compute power, but do not affect the output. 

**Pruning** is the process of setting these weak weights to exactly zero and deleting them from the network.

```
         Unstructured Pruning                   Structured Pruning
      (Remove random weak links)           (Delete entire attention heads)
      
          ○      ○      ○                      ○            ○
         / \    / \    / \                    / \          / \
        ○   x  ○   ○  x   ○                  ○   ○        ○   ○
         \ /    \ /    \ /                    \ /          \ /
          ○      ○      ○                      ○            ○
```

1.  **Unstructured Pruning:** You identify individual connections anywhere in the network that are below a certain strength threshold and remove them. This creates a "sparse" network. While it reduces parameters, standard GPUs are not optimized for sparse matrices, so this rarely yields speedups without specialized hardware.
2.  **Structured Pruning:** You remove entire layers, channels, or attention heads. Because it maintains clean, dense blocks of matrices, the pruned model runs immediately faster on standard hardware.

---

## Mixture of Experts (MoE) — The Specialist Panel

### Don't wake up the whole brain
In a standard LLM, every single token you input passes through every single layer and parameter. If you ask a 100-billion-parameter model: *"What is 2 + 2?"*, it activates all 100 billion parameters to answer. This is highly inefficient.

**Mixture of Experts (MoE)** splits the feed-forward layers of the Transformer into multiple independent **"experts"** (e.g., 8 different mini-neural networks). A **Router network** reads the incoming token and decides which 2 experts are best suited to process it.

```
                       [ Incoming Token ]
                               │
                               ▼
                       [ Router Network ]
                         /            \
               (Active) /              \ (Inactive)
                       ▼                ▼
                  [Expert A]        [Expert B]
                  Math & Code       Literature
                       \                /
                        ▼              ▼
                     [ Combined Output ]
```

By activating only 2 out of 8 experts per token, a model with 47 billion total parameters (like Mixtral 8x7B) only uses **13 billion parameters** of active compute per token. You get the intelligence of a massive model with the speed and cost of a much smaller one.

---

## DPO — Direct Preference Optimization

### Simpler Alignment
As seen in Part 1, standard RLHF is highly complex. You have to train a separate Reward Model, then run a reinforcement learning loop (PPO) that updates the LLM based on reward scores. PPO is notoriously unstable—the model can cheat the reward model (reward hacking) or crash during training.

**DPO (Direct Preference Optimization)**, introduced in 2023, bypasses PPO entirely. It proves mathematically that you can optimize the LLM directly on human preferences without a separate reward model. 

```
  RLHF:  LLM ──► [Reward Model] ──► [PPO Optimizer] ──► [Aligned LLM] (Complex)
  
  DPO:   LLM ──► [Chosen vs Rejected Pairs] ──► [DPO Loss] ──► [Aligned LLM] (Simple)
```

You feed the model pairs of responses: a **Chosen** response (helpful, accurate) and a **Rejected** response (toxic, incorrect) for the same prompt. The DPO loss function increases the probability of the chosen response while simultaneously decreasing the probability of the rejected response in a single supervised training step.

---

## Speculative Decoding — The Speed Writer

### Draft and Verify
LLM inference is slow because it is **memory-bandwidth bound**: the GPU has to load all 70 billion parameters from its memory to chip cache just to generate a single token, then repeat the process for the next.

**Speculative Decoding** speeds this up by using a small, ultra-fast "draft" model (e.g., LLaMA-1.1B) alongside our giant "target" model (LLaMA-70B).

```
  1. Draft Model (Fast):   Generates: "The cat sat on the" (5 tokens)
                              │
  2. Target Model (Slow):  Verifies all 5 tokens in parallel in a single step.
                              │
  3. Action:               If 4 are correct: Keep 4, discard 1, generate new.
```

1.  The small draft model quickly generates 5 candidate tokens in sequence (which takes almost no time).
2.  The large target model evaluates all 5 tokens **in parallel in a single GPU step** (since verification of a sequence can be run in parallel, unlike sequential generation).
3.  If the large model agrees with the draft model's guesses, we accept all 5 tokens instantly. If it disagrees at token 4, we keep the first 3 and discard the rest.

This yields a **2x to 3x speedup** in interactive generation speeds, with mathematically identical output quality.

---

## Complete Efficiency Comparison

| Technique | Memory Savings | Speedup | Quality Impact | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **LoRA** | 4x to 8x (Training) | None (Inference) | Minimal (<1%) | Fine-tuning on a budget. |
| **QLoRA** | 16x (Training) | Slightly Slower | Small (<3%) | Fine-tuning on consumer GPUs. |
| **Quantization (INT4)** | 4x (Inference) | 2x to 4x | Small to Medium | Local serving on laptops/edge devices. |
| **Distillation** | Permanent (Smaller model) | Permanent | Medium (depends on size) | Creating fast, cheap task specialists. |
| **Pruning (Structured)** | 1.5x to 2x | 1.5x to 2x | Small | Shrinking production serving models. |
| **Mixture of Experts** | 4x active compute | 3x to 4x | Often improves quality | Massive models needing fast inference. |
| **Speculative Decoding** | None | 2x to 3x | None (Identical) | Improving user chat generation speeds. |

---

## Common Questions & Answers

**Q: Should I use LoRA or full fine-tuning?**  
**A:** LoRA is the default choice for almost all enterprise use cases. Full fine-tuning is only necessary when the model needs to learn a completely new language, a massive body of raw knowledge, or a highly specific, complex syntax that isn't present in the base model. For style, formatting, and task alignment, LoRA matches full fine-tuning at a fraction of the cost.

**Q: What is the difference between QLoRA and quantization formats like GGUF?**  
**A:** QLoRA is a **training** technique (quantizing the base model to save memory while computing gradients for adapters). GGUF, GPTQ, and AWQ are **inference** formats (compressing a model's weights permanently after all training is finished so that it runs fast and fits on consumer hardware).

**Q: If Mixture of Experts is so efficient, why isn't every model an MoE?**  
**A:** While MoE models require less active compute per token, the entire model (e.g., all 47B parameters) must still be loaded into VRAM memory. Thus, the VRAM memory footprint of an MoE is still that of a giant model, making it difficult to host on low-VRAM servers.

---

---

# Advanced RAG Techniques

> **Context:** You already know the basic RAG pipeline — embed query → search vector DB → stuff chunks into prompt → generate answer. This chapter covers the production-grade techniques that make RAG actually reliable.

---

## The Problem With Basic RAG

Basic RAG is easy to build in a weekend, but fails in production because of structural bottlenecks:

```
  Query: "What was the revenue impact of the 2023 European warehouse fire?"
  
  [Failure 1: Vocabulary Mismatch] ──► User says "fire", document says "thermal incident".
                                       Vector search similarity score: LOW. Misses doc.
                                       
  [Failure 2: Lack of Context]      ──► Chunks are split blindly. Retargeted chunk says:
                                       "It caused a loss of €12M." (Doesn't mention fire).
                                       
  [Failure 3: Lost in the Middle]   ──► Search retrieves 20 docs. The actual answer is
                                       in Doc 11. LLM ignores the middle context.
                                       
  [Failure 4: Single-Step limit]    ──► Query requires comparing 2022 vs 2023 fires.
                                       One-shot retrieval cannot merge time periods.
```

The advanced techniques below solve these specific failure modes.

---

## Hybrid Search — Synonyms & Exact Matches

### The Concept Searcher vs. The Filer
*   **Vector Search (Semantic):** Great at conceptual matches. It knows that "dog" is similar to "canine" and "puppy". But it can easily miss exact matching terms, product codes, or serial numbers (e.g., *"SKU-99214-A"*).
*   **BM25 Search (Keyword):** A classical term-matching algorithm. It does not understand synonyms, but it is incredibly precise at finding exact words, typos, and serial codes.

**Hybrid Search** runs both retrievers in parallel and merges their rankings using an algorithm called **Reciprocal Rank Fusion (RRF)**.

```
  Query: "SKU-4091 insulin therapy"
  
  Vector Search:   1. Diabetes management... (score 0.88)
                   2. Hormonal therapy...   (score 0.82)
                   
  BM25 Search:     1. Instruction manual for SKU-4091... (score 19.4)  ◄── Match!
  
  RRF Fusion:      1. Instruction manual for SKU-4091...
                   2. Diabetes management...
```

### Reciprocal Rank Fusion (RRF)
RRF scores each document based on its rank in both systems, giving higher priority to items that appear near the top of either list.

```python
def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """
    Combines search rankings from vector and keyword searchers.
    k: A constant scaling factor (60 is standard) to control weight of low ranks.
    """
    scores = {}
    
    # Process vector ranks
    for rank, doc_id in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    # Process keyword ranks
    for rank, doc_id in enumerate(keyword_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    # Sort documents by their combined fusion score
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

## Metadata Filtering — Sorting Before Searching

### The Filing Cabinet Drawer
If your organization has 100,000 files spanning five years across 20 countries, searching the entire database for a query like *"What was the German marketing spend in Q3 2023?"* is a recipe for retrieval failure. 

Instead of searching everything, we attach metadata tags (`country: Germany`, `department: Marketing`, `year: 2023`, `quarter: Q3`) to our chunks. At query time, we **filter the filing cabinet drawers first**, and run our vector search only on the matching files.

```
  [User Query] ──► (LLM extracts tags) ──► Apply Filter: [Germany, Q3 2023]
                                                   │
                                                   ▼
                                         [Narrow subset of docs]
                                                   │
                                                   ▼
                                         (Run Semantic Search)
```

### Self-Querying Retrieval
Using a structured query constructor, we can ask an LLM to read the user's natural language question, extract the filter tags, and generate the filter JSON automatically.

```python
# Conceptual metadata filtering structure
metadata_filters = {
    "region": "Germany",
    "department": "Marketing",
    "year": 2023,
    "quarter": "Q3"
}

# The vector database applies these metadata rules first
# filtering out 99% of irrelevant documents before computing similarity
results = vector_db.similarity_search(
    query="spend details on digital ads",
    filter=metadata_filters,
    k=3
)
```

---

## Parent-Child Retrieval — The Summary Index

### The Chunking Dilemma
*   If chunks are **too small** (e.g., a single sentence), the embedding is highly precise. However, the retrieved sentence passed to the LLM lacks context: *"The growth rate was 14%."* (14% of what? In what country?).
*   If chunks are **too large** (e.g., 3 pages), the context is rich, but the semantic representation gets diluted. A paragraph about fire safety buried in 3 pages of building codes will have a low similarity score.

**Parent-Child Retrieval** splits the document into large **Parent Chunks** (e.g., a 2,000-token page) and smaller **Child Chunks** (e.g., 400-token paragraphs nested inside).

```
         Parent Document (2000 tokens)
         ┌───────────────────────────┐
         │                           │
         └───────────────────────────┘
          /            │            \
         ▼             ▼             ▼
     Child A        Child B       Child C     (400 tokens each)
     (Embedded)    (Embedded)    (Embedded)
```

1.  We embed and search only the small **Child Chunks** to locate the exact topic.
2.  Once a match is found (e.g., Child B), we look up its parent ID.
3.  We pull the large **Parent Chunk** from memory and feed it to the LLM. 

This gives us the best of both worlds: high-precision retrieval matching coupled with comprehensive contextual generation.

---

## Multi-Step Querying & HyDE

### HyDE (Hypothetical Document Embedding)
When a user asks a short question like *"How do I fix a leaking valve?"*, the embedding coordinates of the question (which is short and questioning) may not align well with the coordinates of the answer (which is long, descriptive, and technical).

**HyDE** solves this by asking the LLM to write a **fake, hypothetical answer** first:
1.  *"Question: How do I fix a leaking valve? LLM: Generate a hypothetical instruction manual page..."*
2.  We take that fake, highly technical manual page, embed it, and use *its* embedding vector to search the database. 
3.  Because the fake answer uses the same vocabulary and tone as the real documents, it matches the correct files with significantly higher accuracy.

```
  [User Query] ──► (LLM: Write Fake Answer) ──► [Fake Text] ──► (Embed & Search) ──► [Real Docs]
```

### Query Decomposition
For complex, multi-part questions like *"Did we make more profit in Europe in 2022 or 2023?"*, the system decomposes the prompt into sub-queries:
1.  *Sub-query 1: "European profit 2022"*
2.  *Sub-query 2: "European profit 2023"*
3.  The system retrieves documents for both sub-queries separately, merges the outputs, and presents the combined data to the LLM.

---

## Graph RAG — The Knowledge Net

### Reconnecting the Chunks
Standard RAG treats documents as a pile of disconnected paper shreds. But facts are interconnected. If page 1 says *"Alice is the lead engineer of Project Apollo"*, and page 40 says *"Project Apollo went over budget"*, a query asking *"Who runs the project that went over budget?"* will fail.

**Graph RAG** uses an LLM to scan documents, extract key entities (people, places, projects) and the relationships between them, and build a **Knowledge Graph**.

```
  (Alice Johnson) ───[ LEADS ]───► (Project Apollo) ───[ STATUS ]───► (Over Budget)
```

By traversing the links (edges) of the graph at query time, the system can connect facts across dozens of documents, answering multi-hop relational questions that are impossible for flat vector databases.

---

## RAG Design Space Comparison

| Strategy | Failure Solved | Complexity | API Cost |
| :--- | :--- | :--- | :--- |
| **Hybrid Search** | Vocabulary mismatch, exact codes. | Low | None |
| **Metadata Filtering** | Out-of-scope document contamination. | Medium | Low (tag extraction) |
| **Parent-Child** | Lack of context vs. poor similarity. | Medium | None |
| **HyDE** | Question-to-answer semantic gap. | Medium | Low (1 LLM call) |
| **Query Decomposition** | Multi-part comparison queries. | High | Medium (multiple calls) |
| **Reranking** | Irrelevant chunks in top-k context. | Medium | Low (Reranker model API) |
| **Graph RAG** | Multi-hop relationship queries. | Very High | High (graph creation) |

---

---

# Reasoning Models & Test-Time Compute

## Thinking Before Speaking

Most traditional LLMs (like GPT-4o) generate tokens on the fly, calculating the most likely next word instantly. They do not have an internal planning space; they are forced to "think" while they are outputting text. If a model starts writing a complex mathematical proof and realizes it made an error on line 2, it cannot backtrack—it must continue writing anyway.

**Reasoning Models** (such as OpenAI's **o1** and DeepSeek's **R1**) introduce **Test-Time Compute**. 

Instead of answering instantly, they are given extra time (and compute budget) at inference time to generate a hidden internal chain of thought, plan their execution, explore different pathways, and self-correct when they hit dead ends before outputting their final response.

```
  Standard LLM:   [Prompt] ───────────────────────────────► [Final Answer] (Immediate)
  
  Reasoning LLM:  [Prompt] ──► [Thinking tokens...] ────► [Final Answer] (Delayed)
                                - Plan approach
                                - Check equations
                                - Correct errors
```

---

## How Chain-of-Thought (CoT) Works

The foundation of reasoning models was the discovery that forcing a model to "show its work" improves accuracy.
*   **Without CoT:** *"Question: [Complex Math]. Answer: 42"* (Usually incorrect, because the model had to guess the output in one forward pass).
*   **With CoT:** *"Question: [Complex Math]. Let's think step by step: Step 1... Step 2... Therefore, the answer is 42."* (Usually correct, because the intermediate tokens act as a "scratchpad" memory).

### OpenAI o1 & DeepSeek-R1
Modern reasoning models automate and scale this scratchpad behavior:
1.  **Reinforcement Learning on Chain of Thought:** Models are trained using RL, rewarded purely for getting the correct final answer on math, logic, and coding benchmarks.
2.  **Emergent Behaviors:** During training, the models spontaneously learn to generate thinking tokens like *"Wait, let me recalculate that..."*, *"If I assume X, then Y fails..."*, or *"Let me try a different approach."* They learn to debug their own logic internally before writing the output visible to the user.

---

## Synthetic Data — The AI Bootstrap

### The Data Drought
By 2024, AI models had consumed almost all high-quality public text on the internet. To continue training larger models, researchers hit a wall: there was no more new text to read. The solution is **Synthetic Data Generation**—using capable LLMs to generate high-quality training data to train the next generation of models.

```
  [Pretrained Genius] ──► (Generate 100,000 Coding Challenges) ──► [New Student Model]
                             (Self-filter for correctness)          (Trained to code)
```

### Self-Correction & Verification
To ensure synthetic data is high-quality (and doesn't propagate errors), we use verification loops:
1.  **Code generation:** An LLM generates 10,000 python coding exercises and matching solutions.
2.  **Verifiable execution:** A python compiler actually runs the generated code. If the code throws an error or fails the test cases, it is discarded immediately.
3.  **Outcome-based filtering:** Only the verified, correct code-answer pairs are kept. 

This high-quality, filtered dataset is then used to train smaller, more efficient student models, a process that cost Stanford researchers only $600 to turn LLaMA into the capable **Alpaca** instruction follower.

---

## Evaluation Metrics — LLMs & Search

To measure if our compressed models, RAG pipelines, or ranking systems are working, we use specialized metrics:

---

### Perplexity — How Confused Is the Model?
**Perplexity (PPL)** measures how "surprised" a language model is by a test dataset. It is mathematically calculated as the exponentiated average cross-entropy loss:

$$\text{PPL} = e^{\text{Loss}}$$

*   **PPL = 1:** The model is 100% confident and predicts the correct word every time.
*   **PPL = 50:** The model is as confused as if it had to choose uniformly between 50 equally likely words at each step.
*   *Lower perplexity is better.* It proves the model understands grammar, facts, and structure.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    # Pass inputs and calculate target loss
    loss = model(**inputs, labels=inputs["input_ids"]).loss

# Compute perplexity
perplexity = torch.exp(loss).item()
print(f"Perplexity score: {perplexity:.2f}")
```

---

### Ranking Metrics — NDCG

When building search or RAG systems, we want the most relevant documents placed at the very top of our results list. **NDCG (Normalized Discounted Cumulative Gain)** measures this, discounting results that are buried deep down the list.

$$\text{DCG}_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$$

*   **Relevance ($rel_i$):** A score indicating how relevant document $i$ is (e.g., 3 = perfect match, 0 = irrelevant).
*   **Discounting ($\log_2(i+1)$):** The denominator grows larger as the rank index $i$ increases, heavily penalizing relevant documents that are placed at the bottom of the list.
*   **NDCG:** The ratio between your model's DCG and the ideal DCG (perfectly sorted). It ranges from 0.0 to 1.0, where 1.0 is a perfect ranking.

---

## Quick Reference & Common Questions — LLMs

### The Big Picture in Plain English
LLMs represent a shift from programming with rules to steering with language. By training a model to autocomplete the internet, it becomes a mirror of human thought. By adding RLHF, alignment, RAG, and reasoning loops, we turn this raw autocomplete mirror into an agent capable of thinking, searching, and problem-solving.

---

### Module 6 Q&A

**Q: What is the "Lost in the Middle" effect?**  
**A:** It is a context retrieval bias. LLMs pay high attention to text at the very beginning and very end of their prompt context window, but frequently ignore or miss details placed in the middle. When building RAG pipelines, we must place the most critical retrieved pages at the start or end of the context, rather than sorting them strictly by database score.

**Q: Why is perplexity a limited metric for chatbots?**  
**A:** Perplexity measures grammatical fluency and predictability, not helpfulness or truth. A model can write extremely fluent, logical-sounding paragraphs that are completely hallucinated and false. That model would have a very low (good) perplexity score, but be highly toxic or useless as a real assistant.

**Q: How does Speculative Decoding speed up generation without losing quality?**  
**A:** Because the verification step in a Transformer is parallelized. It takes the same amount of time for a giant GPU to evaluate 5 tokens at once as it does to generate 1 token. By letting a tiny model write a draft and having the large model verify all draft tokens in a single parallel step, we get the exact outputs of the large model up to 3x faster.