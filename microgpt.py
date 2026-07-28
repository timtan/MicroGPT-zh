"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import argparse
import hashlib
import math     # math.log, math.exp
import os
import pickle
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

parser = argparse.ArgumentParser(description="Train or sample the tiny GPT")
parser.add_argument('--steps', type=int, default=5000, help="total training steps")
parser.add_argument('--resume', action='store_true', help="resume an interrupted training run")
parser.add_argument('--inference', action='store_true', help="load the checkpoint without training")
parser.add_argument('--checkpoint', default='checkpoint.pkl', help="checkpoint file to save or load")
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top-p', type=float, default=1.0, help="1.0 disables nucleus sampling")
parser.add_argument('--samples', type=int, default=20)
parser.add_argument('--sample-seed', type=int, default=42)
args = parser.parse_args()
if args.steps < 0:
    parser.error('--steps must be non-negative')
if not 0 < args.temperature:
    parser.error('--temperature must be positive')
if not 0 < args.top_p <= 1:
    parser.error('--top-p must be in (0, 1]')
if args.samples < 1:
    parser.error('--samples must be positive')
if args.resume and args.inference:
    parser.error('--resume and --inference cannot be used together')

checkpoint_path = args.checkpoint

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
with open('input.txt', encoding='utf-8') as f:
    docs = [line.strip() for line in f if line.strip()]
assert docs, "input.txt is empty"
dataset_hash = hashlib.sha256('\n'.join(docs).encode()).hexdigest()
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
chars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
stoi = {ch: i for i, ch in enumerate(chars)} # string to integer
itos = {i: ch for i, ch in enumerate(chars)} # integer to string
encode = lambda text: [stoi[ch] for ch in text]
decode = lambda tokens: ''.join(itos[token] for token in tokens)
BOS = len(chars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(chars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 2     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = max(len(doc) for doc in docs) + 1 # longest name, plus one position to predict BOS (end)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer
completed_steps = 0

checkpoint_config = {
    'dataset_hash': dataset_hash,
    'chars': chars,
    'n_layer': n_layer,
    'n_embd': n_embd,
    'block_size': block_size,
    'n_head': n_head,
}

def save_checkpoint(target_steps):
    checkpoint = {
        'version': 1,
        'config': checkpoint_config,
        'target_steps': target_steps,
        'completed_steps': completed_steps,
        'params': [p.data for p in params],
        'm': m,
        'v': v,
        'random_state': random.getstate(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    temporary_path = checkpoint_path + '.tmp'
    with open(temporary_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    os.replace(temporary_path, checkpoint_path)
    print(f"checkpoint saved: {checkpoint_path} ({completed_steps} steps)")

def load_checkpoint():
    global m, v, completed_steps
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    if checkpoint.get('version') != 1 or checkpoint.get('config') != checkpoint_config:
        raise ValueError('checkpoint does not match input.txt or model settings')
    if len(checkpoint['params']) != len(params):
        raise ValueError('checkpoint parameter count does not match the model')
    for p, data in zip(params, checkpoint['params']):
        p.data = data
    m, v = checkpoint['m'], checkpoint['v']
    completed_steps = checkpoint['completed_steps']
    random.setstate(checkpoint['random_state'])
    return checkpoint['target_steps']

def document_loss(doc):
    tokens = [BOS] + encode(doc) + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        probs = softmax(gpt(token_id, pos_id, keys, values))
        losses.append(-probs[target_id].log())
    return (1 / n) * sum(losses)

def evaluate():
    return sum(document_loss(doc).data for doc in docs) / len(docs)

def choose_token(probs, sample_rng):
    weights = [p.data for p in probs]
    token_ids = list(range(vocab_size))
    if args.top_p < 1.0:
        ranked = sorted(zip(token_ids, weights), key=lambda item: item[1], reverse=True)
        kept, cumulative = [], 0.0
        for token_id, weight in ranked:
            kept.append((token_id, weight))
            cumulative += weight
            if cumulative >= args.top_p:
                break
        token_ids, weights = map(list, zip(*kept))
    return sample_rng.choices(token_ids, weights=weights)[0]

def generate_samples():
    sample_rng = random.Random(args.sample_seed)
    samples = []
    for _ in range(args.samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            probs = softmax([l / args.temperature for l in gpt(token_id, pos_id, keys, values)])
            token_id = choose_token(probs, sample_rng)
            if token_id == BOS:
                break
            sample.append(token_id)
        samples.append(decode(sample))
    return samples

def report(include_loss):
    if include_loss:
        print(f"mean training loss: {evaluate():.4f}")
    print(f"--- inference (temperature={args.temperature}, top_p={args.top_p}) ---")
    for sample_idx, sample in enumerate(generate_samples(), 1):
        print(f"sample {sample_idx:2d}: {sample}")

if args.resume or args.inference:
    saved_target_steps = load_checkpoint()
    print(f"checkpoint loaded: {checkpoint_path} ({completed_steps} steps)")
    if args.resume and args.steps != saved_target_steps:
        raise ValueError(f'--steps must remain {saved_target_steps} when resuming this training run')

if not args.inference:
    if args.steps <= completed_steps:
        raise ValueError(f'--steps must be greater than completed steps ({completed_steps})')
    for step in range(completed_steps, args.steps):
        loss = document_loss(docs[step % len(docs)])
        loss.backward()

        lr_t = learning_rate * (1 - step / args.steps) # linear learning rate decay
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        completed_steps = step + 1
        print(f"step {completed_steps:5d} / {args.steps:5d} | loss {loss.data:.4f}", end='\r')
        if completed_steps % 1000 == 0 or completed_steps == args.steps:
            print()
            save_checkpoint(args.steps)
        if completed_steps % 5000 == 0 or completed_steps == args.steps:
            report(include_loss=True)
else:
    report(include_loss=False)
