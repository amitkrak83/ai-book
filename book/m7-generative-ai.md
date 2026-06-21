# Module 7 — Generative AI *(2014 – 2023)*

**Why this era exists:** Every model up to this point was discriminative — it classified, predicted, or extracted. Generative AI flips the direction: instead of mapping input to a label, the model learns the underlying distribution of the data and can *sample new examples* from it.

---

## Generative vs. Discriminative Models

### The Judge vs. The Creator
To understand generative AI, think of two different roles in an art gallery:
*   **The Art Inspector (Discriminative):** A judge who stands in front of a painting, looks at the brushstrokes, and says: *"This is a genuine Rembrandt"* or *"This is a fake"*. They model the boundary between classes ($P(\text{Label} \mid \text{Image})$).
*   **The Painter (Generative):** A creator who sits in front of a blank canvas and paints a brand-new masterpiece that looks like a Rembrandt. To do this, they must understand the entire distribution of how Rembrandt painted—his colors, themes, lighting, and textures ($P(\text{Image})$).

```
  Discriminative: [ Image ] ──► ( Calculate Boundary ) ──► [ Label: Rembrandt ]
  
  Generative:     [ Noise/Seed ] ──► ( Sample from Distribution ) ──► [ New Rembrandt Image ]
```

Discriminative models (like our classifiers in Module 1 and 3) learn to separate data. Generative models learn to *understand the data itself* so deeply that they can create fresh, realistic variations of it.

---

## Variational Autoencoders (VAEs) — The Caricature Sketcher

### The Simplified Code
Imagine a portrait sketcher who sits at a tourist boardwalk. Instead of painting every single pixel of a tourist's face, they quickly write down a simplified set of codes in their notebook: *"male, wearing glasses, wide smile, brown hair, round face"*. This simplified description is called a **latent vector** (or bottleneck). If you give this description to a second artist who hasn't seen the tourist, they can read the notes and paint a highly realistic face that looks exactly like the person.

This is what an **Autoencoder** does:
1.  **The Encoder:** Compresses a high-dimensional image (e.g., $256 \times 256$ pixels) into a small list of numbers representing key features (latent space).
2.  **The Decoder:** Takes those numbers and reconstructs the full-size image.

```
  [Input Image] ──► [ Encoder ] ──► [ Latent Bottleneck (Z) ] ──► [ Decoder ] ──► [ Reconstructed Image]
```

### The Variational Breakthrough
Traditional autoencoders had a fatal flaw: the encoder mapped images to single, isolated points in space. If you chose a random point in between those points, the decoder outputted scrambled, terrifying static because it had never seen that coordinate before. The space had **"holes"**.

**Variational Autoencoders (VAEs)** fixed this by forcing the encoder to output a **probability distribution** (a mean $\mu$ and a variance $\sigma$) instead of a single point. 

By adding a mathematical constraint (called Kullback-Leibler Divergence) that forces these distributions to overlap and cluster around a standard normal distribution, the latent space becomes smooth and continuous. You can now choose *any* coordinate on the map, and the decoder is guaranteed to draw a smooth, logical face.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, img_dim=784, latent_dim=20):
        super().__init__()
        # Encoder projects image to mean and log-variance layers
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
        # Reparameterization trick: samples noise, scales by variance, adds mean.
        # This keeps the sampling step differentiable so backprop works!
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (did we draw the image correctly?)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence loss (is our coordinate space smooth and standard?)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## GANs — The Counterfeiter and the Detective

### Pitting Two Brains Against Each Other
In 2014, Ian Goodfellow introduced **Generative Adversarial Networks (GANs)**. Before GANs, generative outputs were blurry. Goodfellow's breakthrough was to train two neural networks simultaneously through competition:
*   **The Counterfeiter (Generator):** Attempts to print fake $100 bills (generate realistic images from random noise). It never gets to look at real bills; it only learns from the detective's feedback.
*   **The Detective (Discriminator):** Receives a mix of real $100 bills and the counterfeiter's fakes. It tries to classify them as "real" or "fake".

```
  [Noise (Z)] ──► [ Generator ] ──► [ Fake Image ] ──┐
                                                     ▼
  [Real Images] ──────────────────────────────► [ Discriminator ] ──► [ Real/Fake Score ]
```

As the detective gets better at spotting the tells of a fake, the counterfeiter is forced to refine its technique. They play a mathematical minimax game:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Eventually, they reach a balance: the counterfeiter creates images so photorealistic that the detective has to guess randomly (50% accuracy).

### Training Code
Here is a single training step for a GAN:

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

def train_step(real_images):
    batch_size = real_images.size(0)
    
    # --- Step 1: Train Discriminator ---
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
    
    # --- Step 2: Train Generator ---
    opt_G.zero_grad()
    # Generator wants discriminator to think fakes are real (targets are 1s!)
    loss_G = criterion(D(fake_images), torch.ones(batch_size, 1))
    loss_G.backward()
    opt_G.step()
```

### Why GANs Stalled (Mode Collapse)
GANs are notoriously unstable to train. The most common failure mode is **Mode Collapse**: the generator finds a single "trick"—like drawing a highly realistic cat in the top-left corner—that always fools the discriminator. 

Instead of learning to draw different cats, dogs, and landscapes, the generator collapses, producing that exact same cat image over and over. If the discriminator adapts, the model oscillates wildly without ever converging.

---

## Diffusion Models — Sculpting from Marble

### Chipping Away the Static
Imagine taking a beautiful marble statue of a lion, taking a hammer, and chipping away small flakes of stone. If you do this 1,000 times, the statue becomes a completely unrecognizable pile of gravel and dust. This is the **Forward Process** of a diffusion model: we take a clean image and gradually add random Gaussian noise over $T$ timesteps until it is pure static.

```
  Clean Image (x0) ──► Add Noise (t=1) ──► ... ──► Pure Static (xT)
  
  Clean Image (x0) ◄── Denoise (t=1) ◄── ... ◄── Pure Static (xT)
```

Now, imagine training a master sculptor who can look at a pile of dust at step $t$, guess exactly where the noise was added, and blow away just enough dust to recover the shape at step $t-1$. This is the **Reverse Process**: we train a neural network (typically a U-Net) to predict the noise added at each step, allowing it to "sculpt" a clear, high-resolution image out of pure random noise.

### Denoising Loss
Instead of predicting pixels directly, the model is trained to predict the noise vector that was added to the image:

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

### Guidance Scales
To generate text-to-image outputs (like Stable Diffusion), we inject text embeddings into the denoising model. 

To force the model to listen closely to our text prompt rather than drawing whatever it wants, we use **Classifier-Free Guidance (CFG)**. During generation, we compute the noise prediction with the prompt, and without the prompt, and blend them:

$$\text{Final Noise} = \text{Unconditioned} + s \times (\text{Conditioned} - \text{Unconditioned})$$

A guidance scale $s$ of 7 to 9 acts like a volume knob, forcing the model to strictly align its denoising steps with the details in the text prompt.

---

## LLMs as Generators — The ChatGPT Interface Shift

By 2020, GPT-3 was already a highly capable next-token generator. Yet it was only used by developers via APIs. In November 2022, OpenAI released **ChatGPT**, which reached 100 million users in two months—the fastest product adoption in history. 

Crucially, **the underlying model did not change significantly**. The breakthrough was an interface shift:
1.  **Alignment via RLHF:** OpenAI used reinforcement learning to align next-token predictions with helpful, safe human communication, turning a document completer into a conversational partner.
2.  **Persistent Chat States:** The user-interface wraps the conversation, sending the entire past chat history back to the model with each new prompt. This creates the illusion of persistent "memory," when in reality, the model is just autocompleting the conversation log on every turn.

---

## Multimodality — Interleaving Sensory Tokens

### The Unified Conceptual Space
Traditional systems used separate pipelines: a CNN for images and an LSTM for text. They could not naturally "talk" to each other.

Modern **multimodal models** (like GPT-4V, Gemini, and Claude 3.5) use a single, unified Transformer backbone to process images, text, audio, and video simultaneously.

```
  Text:  "This chart shows: " ──► [ Token Embeddings ] ──┐
                                                          ├──► [ Unified Transformer ]
  Image: [ 224x224 Image ]     ──► [ Patch Projections ] ──┘
```

1.  **Image Slicing:** An input image is sliced into flat patches (e.g., $16 \times 16$ pixels) using a Vision Transformer (ViT).
2.  **Projection:** These patches are projected into vector embeddings that share the exact same dimension size as our text token embeddings.
3.  **Interleaving:** We concatenate the sequence: `[Text Token, Text Token, Image Patch Token, Image Patch Token]`.
4.  **Attention:** The Transformer processes this sequence. Self-attention doesn't care where a token came from; it allows a text token (*"revenue"*) to attend directly to an image patch token (representing a rising bar on a chart).

---

## Prompt Engineering & Evaluations

The same model can yield completely different outputs depending on the prompt framing. 
*   **Zero-shot:** Prompting directly without examples.
*   **Few-shot:** Showing examples of the input-output format.
*   **Chain-of-Thought:** Adding *"think step by step"* to write down intermediate reasoning steps.

### Golden Test Sets & Judges
Because LLM outputs are free text, we cannot evaluate them using simple accuracy metrics. We use:
1.  **Golden Test Sets:** A curated list of 100+ prompt-response test cases. When you update a prompt template, you run the test suite and check for regressions.
2.  **LLM-As-A-Judge:** We feed the output of our model to a larger, stronger LLM (like GPT-4) and ask it to rate the output on a scale of 1 to 5 against a structured rubric.

```python
def llm_judge(model_output, grading_rubric, judge_client):
    """
    Evaluates LLM output quality using a stronger model as the grader.
    """
    prompt = f"""
    You are an impartial grading judge. Evaluate the output text based on this rubric:
    {grading_rubric}
    
    Output to evaluate:
    "{model_output}"
    
    Provide your evaluation score as a single integer from 1 to 5.
    Score:"""
    
    response = judge_client.generate(prompt)
    return int(response.strip())
```

---

## Structured Outputs — The Grammar Gate

### The Formatting Gate
Many applications require JSON data to parse into database fields. If you ask an LLM to output JSON, it might output: *"Here is your JSON: { ... }"*, which breaks code parsers.

To guarantee perfect schema conformance, modern APIs use **Grammar Constraints** (Structured Outputs).

```
  Model Output Step ──► [ Grammar Conformance Check ] ──► Allowed Token Selection
                           Must fit Pydantic schema
```

Instead of letting the model choose from its entire 100,000-word vocabulary at every step, the API parser compiles your target JSON schema into a set of state-machine rules. At each token selection step:
*   If the schema demands a key: the model is forbidden from selecting anything except text within quotes.
*   If the schema demands a number: only digit tokens are allowed to fire.

This acts as a **grammar gate** that forces the model to generate 100% valid, conforming JSON structures with zero syntax errors.

---

## Quick Reference & Common Questions — Generative AI

### The Big Picture in Plain English
Discriminative models draw boundaries to separate data (e.g., drawing a line between cat photos and dog photos). Generative models map the entire high-dimensional layout of the data so they can draw new, unique samples that fit the pattern.

---

### Module 7 Q&A

**Q: What is mode collapse in GANs?**  
**A:** Mode collapse occurs when the generator discovers a single fake image that consistently fools the discriminator (e.g., one realistic portrait). Instead of learning to generate a diverse range of faces, the generator collapses and outputs that exact same image every time, rendering the model useless.

**Q: Why are diffusion models more stable to train than GANs?**  
**A:** GANs rely on an unstable adversarial minimax game where both networks are constantly chasing a moving target. Diffusion models, on the other hand, train on a simple, stable Mean Squared Error loss: they are given a noisy image and simply learn to predict the added noise, making optimization highly stable.

**Q: What is Classifier-Free Guidance (CFG) in text-to-image generators?**  
**A:** CFG is a technique that controls how closely the generator adheres to your text prompt. By blending the noise predictions of a conditioned pass (with the prompt) and an unconditioned pass (without the prompt), we can amplify the text signal, guiding the denoising process to match the prompt details.