# Module 7 — Generative AI *(2014 – 2023)*

**Why this era exists:** Every model up to this point in the lineage (from Linear Regression to ResNet) was discriminative — it classified, predicted, or extracted. It learned the boundary between things. Generative AI flips the direction: instead of mapping an input to a label, the model learns the underlying probability distribution of the data itself, allowing it to *sample new, never-before-seen examples* from that distribution.

---

## Generative vs. Discriminative Models

### Why
Traditional models answer the question: *"Given this data, what is the label?"* ($P(Y \mid X)$). Generative models answer: *"What does data with this label look like?"* ($P(X \mid Y)$ or simply $P(X)$). To generate, a model must possess a vastly deeper understanding of the world. It cannot just find a shortcut to separate two classes; it must understand the anatomy of the classes themselves.

### What: The Judge vs. The Creator Analogy
To understand the shift, think of two different professions in an art gallery:
*   **The Art Inspector (Discriminative):** A judge who stands in front of a painting, looks at the brushstrokes, and says: *"This is a genuine Rembrandt"* or *"This is a fake"*. They only need to know the distinguishing boundary features.
*   **The Painter (Generative):** A creator who sits in front of a blank canvas and paints a brand-new masterpiece that looks like a Rembrandt. To do this, they must understand the entire distribution of how Rembrandt painted—his colors, themes, lighting, and textures.

```text
  Discriminative: [ Image ] ──► ( Calculate Boundary ) ──► [ Label: Rembrandt ]
  
  Generative:     [ Noise/Seed ] ──► ( Sample from Distribution ) ──► [ New Rembrandt Image ]
```

---

## Autoencoders & VAEs — The Caricature Sketcher *(2013)*

### Why
Before we can generate complex data like images, we must find a way to compress high-dimensional pixels into a smaller, meaningful representation. A $256 \times 256$ RGB image has nearly 200,000 pixels. Most random combinations of those pixels are just TV static. The subset of pixels that form "human faces" lies on a much lower-dimensional manifold.

### What
An **Autoencoder** is a neural network trained to copy its input to its output, but it is forced through a tiny bottleneck layer.
1.  **The Encoder:** Compresses the high-dimensional image into a small list of numbers representing key features (the **latent vector**).
2.  **The Decoder:** Takes those numbers and reconstructs the full-size image.

Think of a portrait sketcher at a tourist boardwalk. Instead of painting every single pixel of a tourist's face, they quickly write down a simplified set of codes in their notebook: *"male, wearing glasses, wide smile, brown hair, round face"*. This is the latent vector. If they pass this notebook to a second artist who hasn't seen the tourist, the second artist (the decoder) can paint a highly realistic face that matches the description.

### The Variational Breakthrough (VAEs)
Traditional autoencoders had a fatal flaw for generation: the encoder mapped images to single, isolated, deterministic points in space. If you chose a random point in between those known points, the decoder would output terrifying, scrambled static. The space had **"holes"**.

**Variational Autoencoders (VAEs)** (Kingma & Welling, 2013) fixed this by forcing the encoder to output a **probability distribution** (a mean $\mu$ and a variance $\sigma$) instead of a single point. 

By adding a mathematical constraint (Kullback-Leibler Divergence) that forces these distributions to overlap and cluster around a standard normal distribution, the latent space becomes smooth and continuous. It is like turning a jagged landscape into a smooth hill. You can now choose *any* coordinate on the map, and the decoder is guaranteed to draw a smooth, logical face. Interpolating between two points (e.g., from "smiling" to "frowning") yields a smooth transition.

### How & Code
The biggest technical hurdle in VAEs was backpropagation. You cannot backpropagate gradients through a random sampling node. The solution is the **Reparameterization Trick**: instead of sampling $Z$ directly from $\mathcal{N}(\mu, \sigma^2)$, we sample a random noise $\epsilon$ from a standard normal $\mathcal{N}(0, 1)$, and then compute $Z = \mu + \sigma \odot \epsilon$. The randomness is pushed to a side input, allowing gradients to flow back through $\mu$ and $\sigma$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_dim=784, latent_dim=20):
        super().__init__()
        # Encoder projects image to mean and log-variance
        self.encoder_fc = nn.Linear(img_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder projects latent vector back to image space
        self.decoder_fc1 = nn.Linear(latent_dim, 400)
        self.decoder_fc2 = nn.Linear(400, img_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # The Reparameterization Trick
        # std is exponentiated half log variance
        std = torch.exp(0.5 * logvar)
        # sample eps from standard normal
        eps = torch.randn_like(std)
        # return z = mu + eps * std
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # 1. Reconstruction loss (did we draw the image correctly? usually MSE or BCE)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. KL Divergence loss (is our coordinate space smooth and standard?)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## GANs — The Counterfeiter and the Detective *(2014)*

### Why
VAEs generate blurry images. Why? Because their reconstruction loss (like Mean Squared Error) penalizes pixel-wise differences. If the model is uncertain about where an edge should be, it averages the possibilities, resulting in a blur. Ian Goodfellow realized we needed a new loss function — one that didn't just compare pixels, but judged *realism*.

### What: Pitting Two Brains Against Each Other
**Generative Adversarial Networks (GANs)** solve the blurriness by training two neural networks simultaneously through competition:
*   **The Counterfeiter (Generator, G):** Attempts to print fake $100 bills (generate realistic images from random noise). It never gets to look at real bills; it only learns from the detective's feedback.
*   **The Detective (Discriminator, D):** Receives a mix of real $100 bills and the counterfeiter's fakes. It tries to classify them as "real" or "fake".

```text
  [Noise (Z)] ──► [ Generator ] ──► [ Fake Image ] ──┐
                                                     ▼
  [Real Images] ──────────────────────────────► [ Discriminator ] ──► [ Real/Fake Score ]
```

As the detective gets better at spotting the tells of a fake, the counterfeiter is forced to refine its technique to fool the detective. They play a mathematical minimax game. When the game reaches equilibrium, the generator creates images so photorealistic that the discriminator has to guess randomly (50% accuracy).

### How & Code
The discriminator wants to maximize the probability of assigning the correct label to both real and fake images. The generator wants to minimize the discriminator's accuracy on fakes.

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

```python
import torch
import torch.nn as nn

# Generator: maps random noise to image dimensions
G = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
# Discriminator: maps image to a realness probability score (0 to 1)
D = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
criterion = nn.BCELoss()

def train_gan_step(real_images):
    batch_size = real_images.size(0)
    
    # ─── Step 1: Train Discriminator ───
    opt_D.zero_grad()
    # Score real images (targets are 1s)
    loss_real = criterion(D(real_images), torch.ones(batch_size, 1))
    
    # Generate fake images and score them (targets are 0s)
    noise = torch.randn(batch_size, 100)
    fake_images = G(noise)
    loss_fake = criterion(D(fake_images.detach()), torch.zeros(batch_size, 1))
    
    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()
    
    # ─── Step 2: Train Generator ───
    opt_G.zero_grad()
    # Generator wants discriminator to think fakes are real (targets are 1s!)
    loss_G = criterion(D(fake_images), torch.ones(batch_size, 1))
    loss_G.backward()
    opt_G.step()
```

### The Fatal Flaw: Mode Collapse
GANs produce incredibly sharp images (like StyleGAN's photorealistic faces), but they are notoriously unstable. The most common failure mode is **Mode Collapse**: the generator finds a single "trick"—like drawing one highly realistic cat—that always fools the discriminator. 

Instead of learning the full distribution (different cats, dogs, landscapes), the generator collapses, producing that exact same cat image over and over. Imagine the counterfeiter found a single flawless $100 bill design and just prints it infinitely. This lack of diversity, combined with training instability, led researchers to seek a new paradigm.

---

## Diffusion Models — Sculpting from Marble *(2020)*

### Why
GANs lacked diversity and stability. VAEs were stable but blurry. Diffusion models (like Midjourney, DALL-E, Stable Diffusion) achieved the holy grail: extreme photorealism, massive diversity, and stable training.

### What: Chipping Away the Static
Imagine a beautiful marble statue of a lion. You take a hammer and chip away small flakes of stone. If you do this 1,000 times, the statue becomes a completely unrecognizable pile of gravel and dust. This is the **Forward Process** of a diffusion model: we take a clean image and gradually add random Gaussian noise over $T$ timesteps until it is pure TV static.

```text
  Clean Image (x_0) ──► Add Noise (t=1) ──► ... ──► Pure Static (x_T)
  
  Clean Image (x_0) ◄── Denoise (t=1)   ◄── ... ◄── Pure Static (x_T)
```

Now, imagine training a master sculptor who can look at a pile of dust at step $t$, guess exactly where the noise was added, and blow away just enough dust to recover the shape at step $t-1$. This is the **Reverse Process**. We train a neural network (typically a U-Net) to predict the noise added at each step, allowing it to "sculpt" a clear, high-resolution image out of pure random noise by running the reverse steps.

### How & Code
Instead of predicting pixels directly, the model is trained to predict the *noise vector* that was added to the image. This makes the math incredibly elegant: it's just Mean Squared Error between the predicted noise and the actual noise we added.

```python
import torch.nn.functional as F

def diffusion_loss(noise_model, x_0, t, noise, alphas_cumprod):
    """
    Computes loss for training a denoising model.
    x_0: Clean input image.
    t: Selected random timestep in the diffusion process.
    noise: The target random noise added.
    alphas_cumprod: Cumulative noise scheduling factors.
    """
    # Step 1: Scale clean image and add noise according to timestep schedule
    sqrt_alpha_prod = alphas_cumprod[t].sqrt()
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]).sqrt()
    
    # Create the noisy image at timestep t
    x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
    
    # Step 2: The model predicts the noise vector based on noisy image and timestep
    predicted_noise = noise_model(x_t, t)
    
    # Step 3: Train model to minimize error between guessed noise and actual noise
    return F.mse_loss(predicted_noise, noise)
```

### Classifier-Free Guidance (CFG): The Volume Knob
To generate text-to-image outputs, we inject text embeddings (from an encoder like CLIP) into the U-Net. 

But how do we force the model to listen closely to our text prompt rather than drawing whatever it wants? We use **Classifier-Free Guidance (CFG)**. During generation, we compute the noise prediction *with* the prompt, and *without* the prompt (using a blank string), and blend them:

$$\text{Final Noise} = \text{Unconditioned} + s \times (\text{Conditioned} - \text{Unconditioned})$$

The guidance scale $s$ acts like a volume knob. A scale of 1 means normal conditioning. A scale of 7 to 9 heavily extrapolates the difference, strictly forcing the model to align its denoising steps with the details in the text prompt.

---

## Multimodality — Interleaving Sensory Tokens

### Why
Humans don't just learn from text. We see, hear, and read simultaneously. Early AI systems used separate, siloed pipelines: a CNN for images, an LSTM for text. They couldn't truly reason about an image in natural language. We needed a unified brain.

### What: The Unified Conceptual Space
Modern **multimodal models** (like GPT-4V, Gemini, Claude 3.5 Sonnet) use a single, unified Transformer backbone to process images, text, audio, and video simultaneously. 

Think of it like a universal translator. Instead of having a "French room" and an "English room", everyone in the UN assembly speaks a single language: **Tokens**. If we can convert images into tokens of the exact same dimensional size as text tokens, the Transformer doesn't care—it just applies self-attention across all of them.

### How
```text
  Text:  "This chart shows: " ──► [ Token Embeddings ] ──┐
                                                         ├──► [ Unified Transformer ]
  Image: [ 224x224 Image ]    ──► [ Patch Projections ] ──┘
```

1.  **Image Slicing (ViT):** An input image is sliced into flat, non-overlapping patches (e.g., $16 \times 16$ pixels).
2.  **Projection:** These patches are passed through a linear layer to project them into vector embeddings that share the *exact same dimension size* as our text embeddings (e.g., 4096 dimensions).
3.  **Interleaving:** We concatenate the sequence into one long context window: `[Text Token, Text Token, Image Patch Token, Image Patch Token]`.
4.  **Cross-Modal Attention:** The Transformer processes this sequence. Self-attention allows a text token (like the word *"revenue"*) to attend directly to an image patch token (representing a rising green bar on a chart). The model inherently learns the mapping between visual concepts and semantic language.

---

## Prompt Engineering & Structured Outputs

### The ChatGPT Interface Shift
By 2020, GPT-3 was a highly capable next-token generator. Yet it was only used by developers. In November 2022, OpenAI released **ChatGPT**, reaching 100 million users in two months. The underlying model didn't change massively. The breakthrough was an interface shift:
1.  **RLHF Alignment:** Turning a document completer into a conversational partner.
2.  **Persistent Chat States:** The UI sends the *entire past chat history* back to the model with each new prompt, creating the illusion of persistent memory.

### Prompting Techniques
Because generative LLMs are highly sensitive to context, how you frame the question changes the distribution it samples from.
*   **Zero-shot:** Prompting directly without examples.
*   **Few-shot:** Showing 2–3 examples of the input-output format to anchor the model's pattern matching.
*   **Chain-of-Thought (CoT):** Adding *"think step by step"*. Think of a math student taking a test. If they just guess the final answer in their head, they make mistakes. If they write out their scratchpad work step-by-step, they arrive at the correct answer. CoT forces the LLM to spend compute tokens on intermediate reasoning before outputting the final answer.

### Structured Outputs — The Grammar Gate
Many applications require JSON data to parse into database fields. If you ask an LLM to output JSON, it might wrap it in markdown: `` ```json { "name": "AI" } ``` ``, breaking your code parsers.

To guarantee perfect schema conformance, modern APIs use **Grammar Constraints** (Structured Outputs).

```text
  Model Output Step ──► [ Grammar Conformance State Machine ] ──► Allowed Token Selection
                           (Must fit strict Pydantic schema)
```

Instead of letting the model choose from its entire 100,000-word vocabulary at every step, the API compiles your target JSON schema into a strict state machine. 
*   If the schema demands a key, the model is physically forbidden from selecting anything except text within quotes.
*   If the schema demands a number, only digit tokens are allowed to have non-zero probability.
This acts like a **grammar gate** that forces the model to generate 100% valid, conforming JSON structures with zero syntax errors.

---

## Quick Reference & Common Questions — Generative AI

### The Big Picture in Plain English
Discriminative models draw boundaries to separate data (e.g., drawing a line between cat photos and dog photos). Generative models map the entire high-dimensional layout of the data so they can draw new, unique samples that fit the pattern. VAEs compress data into a smooth latent space. GANs pit a generator against a discriminator. Diffusion models learn to reverse the process of turning an image into static.

---

### Module 7 Q&A

**Q: What is the Reparameterization Trick and why is it needed in VAEs?**  
**A:** In a VAE, the encoder outputs a mean and variance. To get the latent vector $Z$, we must sample from this distribution. But you cannot backpropagate gradients through a random sampling operation. The reparameterization trick pulls the randomness out into a separate variable $\epsilon \sim \mathcal{N}(0,1)$, computing $Z = \mu + \sigma \cdot \epsilon$. This allows gradients to flow safely through $\mu$ and $\sigma$ back to the encoder.

**Q: What is mode collapse in GANs?**  
**A:** Mode collapse occurs when the generator discovers a single fake image that consistently fools the discriminator (e.g., one ultra-realistic portrait). Instead of learning to generate a diverse range of faces, the generator collapses and outputs that exact same image every time, rendering the model useless for diverse generation.

**Q: Why are diffusion models more stable to train than GANs?**  
**A:** GANs rely on an unstable adversarial minimax game where both networks are constantly chasing a moving target. If one overpowers the other, training collapses. Diffusion models train on a simple, stable Mean Squared Error loss: they are given a noisy image and simply learn to predict the added noise. Optimization is straightforward and doesn't suffer from mode collapse.

**Q: What is Classifier-Free Guidance (CFG) in text-to-image generators?**  
**A:** CFG controls how closely the generator adheres to your text prompt. By evaluating the model twice per step—once conditioned on the text prompt and once unconditioned (blank prompt)—we can find the vector pointing toward the prompt's meaning. We then scale this vector by the guidance scale (e.g., 7.5) to push the denoising process strongly toward the text description.

**Q: How do vision-language models (like GPT-4V) process images?**  
**A:** They don't use separate CNNs. They slice the image into patches (like a grid), project each patch into a dense vector embedding that matches the dimension of text embeddings, and concatenate them with the text tokens. A single unified Transformer then processes the mixed sequence of text and image tokens using cross-modal self-attention.