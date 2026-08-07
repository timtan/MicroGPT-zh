"""
The same atomic GPT trainer and sampler, ported to PyTorch.
torch tensors and autograd replace the pure-Python Value class;
everything else follows microgpt_en.py step for step.
"""

import os
import random
import torch
random.seed(42)  # Let there be order among chaos
torch.manual_seed(42)

# Pick the compute device; override with e.g. MICROGPT_DEVICE=mps (Apple GPU) or cpu.
# Note: this model is so tiny that CPU beats MPS — GPU kernel-launch overhead dominates.
device = torch.device(os.environ.get('MICROGPT_DEVICE', 'cpu'))
print(f"device: {device}")

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs)))  # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars)  # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters, to store the knowledge of the model
# (torch.autograd replaces the hand-written Value autograd from the pure-Python version)
n_layer = 1      # depth of the transformer neural network (number of layers)
n_embd = 16      # width of the network (embedding dimension)
block_size = 16  # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4       # number of attention heads
head_dim = n_embd // n_head  # derived dimension of each head
matrix = lambda nout, nin, std=0.08: torch.nn.Parameter(torch.randn(nout, nin, device=device) * std)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(state_dict.values())
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def rmsnorm(x):
    ms = (x * x).mean(-1, keepdim=True)
    return x * (ms + 1e-5) ** -0.5

def gpt(token_ids):
    # token_ids: (T,) tensor; returns (T, vocab_size) logits with causal attention
    T = token_ids.shape[0]
    tok_emb = state_dict['wte'][token_ids]           # token embeddings (T, n_embd)
    pos_emb = state_dict['wpe'][:T]                  # position embeddings (T, n_embd)
    x = rmsnorm(tok_emb + pos_emb)  # note: not redundant due to backward pass via the residual connection

    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']).view(T, n_head, head_dim).transpose(0, 1)
        k = linear(x, state_dict[f'layer{li}.attn_wk']).view(T, n_head, head_dim).transpose(0, 1)
        v = linear(x, state_dict[f'layer{li}.attn_wv']).view(T, n_head, head_dim).transpose(0, 1)
        attn_logits = (q @ k.transpose(-2, -1)) / head_dim ** 0.5  # (n_head, T, T)
        attn_logits = attn_logits.masked_fill(~causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        x_attn = (attn_weights @ v).transpose(0, 1).reshape(T, n_embd)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo']) + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1']).relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2']) + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

# Repeat in sequence
num_steps = 1000  # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the whole token sequence at once, all the way to the loss
    inputs = torch.tensor(tokens[:n], device=device)
    targets = torch.tensor(tokens[1:n+1], device=device)
    logits = gpt(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)  # average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    optimizer.zero_grad()
    loss.backward()

    # Adam optimizer update with linear learning rate decay
    lr_t = learning_rate * (1 - step / num_steps)
    for group in optimizer.param_groups:
        group['lr'] = lr_t
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
with torch.no_grad():
    for sample_idx in range(20):
        token_ids = [BOS]
        sample = []
        for pos_id in range(block_size):
            logits = gpt(torch.tensor(token_ids, device=device))
            probs = torch.softmax(logits[-1] / temperature, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            if token_id == BOS:
                break
            token_ids.append(token_id)
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
