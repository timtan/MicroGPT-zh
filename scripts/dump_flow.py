"""Train (once) the English microgpt and dump every intermediate vector
for the sequence BOS -> 'e' -> 'm', for the flow.html scrollytelling page.

usage: uv run python scripts/dump_flow.py

- Weights come from checkpoints/microgpt-en-1l-1000-seed42.pkl; if missing,
  replicates microgpt_en.py's training run (seed 42, 1000 steps) using
  data/names_en.txt (never touches input.txt, which holds the Chinese dataset).
- Writes data/flow_dump.json, and if flow.html exists, splices the JSON in
  between the /*__FLOW_DATA_START__*/ ... /*__FLOW_DATA_END__*/ markers.
"""
import json
import math
import os
import pickle
import random
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(ROOT, 'checkpoints', 'microgpt-en-1l-1000-seed42.pkl')
NAMES_PATH = os.path.join(ROOT, 'data', 'names_en.txt')
JSON_PATH = os.path.join(ROOT, 'data', 'flow_dump.json')
HTML_PATH = os.path.join(ROOT, 'flow.html')
NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'

N_LAYER, N_EMBD, BLOCK_SIZE, N_HEAD = 1, 16, 16, 4
HEAD_DIM = N_EMBD // N_HEAD
NUM_STEPS = 1000


def ensure_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, 'rb') as f:
            return pickle.load(f)
    return train_and_save()


def train_and_save():
    # Mirrors microgpt_en.py exactly (seed, shuffle, init order, Adam schedule),
    # so the resulting weights are the ones the talk's code would produce.
    random.seed(42)
    if not os.path.exists(NAMES_PATH):
        os.makedirs(os.path.dirname(NAMES_PATH), exist_ok=True)
        urllib.request.urlretrieve(NAMES_URL, NAMES_PATH)
    docs = [line.strip() for line in open(NAMES_PATH) if line.strip()]
    random.shuffle(docs)
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"training: {len(docs)} docs, vocab {vocab_size}")

    class Value:
        __slots__ = ('data', 'grad', '_children', '_local_grads')

        def __init__(self, data, children=(), local_grads=()):
            self.data = data
            self.grad = 0
            self._children = children
            self._local_grads = local_grads

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data + other.data, (self, other), (1, 1))

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data * other.data, (self, other), (other.data, self.data))

        def __pow__(self, other):
            return Value(self.data**other, (self,), (other * self.data**(other - 1),))

        def log(self):
            return Value(math.log(self.data), (self,), (1 / self.data,))

        def exp(self):
            return Value(math.exp(self.data), (self,), (math.exp(self.data),))

        def relu(self):
            return Value(max(0, self.data), (self,), (float(self.data > 0),))

        def __neg__(self): return self * -1
        def __radd__(self, other): return self + other
        def __sub__(self, other): return self + (-other)
        def __rmul__(self, other): return self * other
        def __truediv__(self, other): return self * other**-1

        def backward(self):
            topo, visited = [], set()

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

    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    state_dict = {'wte': matrix(vocab_size, N_EMBD), 'wpe': matrix(BLOCK_SIZE, N_EMBD), 'lm_head': matrix(vocab_size, N_EMBD)}
    for i in range(N_LAYER):
        state_dict[f'layer{i}.attn_wq'] = matrix(N_EMBD, N_EMBD)
        state_dict[f'layer{i}.attn_wk'] = matrix(N_EMBD, N_EMBD)
        state_dict[f'layer{i}.attn_wv'] = matrix(N_EMBD, N_EMBD)
        state_dict[f'layer{i}.attn_wo'] = matrix(N_EMBD, N_EMBD)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * N_EMBD, N_EMBD)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(N_EMBD, 4 * N_EMBD)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"num params: {len(params)}")

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
        x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]
        x = rmsnorm(x)
        for li in range(N_LAYER):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])
            v = linear(x, state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(N_HEAD):
                hs = h * HEAD_DIM
                q_h = q[hs:hs + HEAD_DIM]
                k_h = [ki[hs:hs + HEAD_DIM] for ki in keys[li]]
                v_h = [vi[hs:hs + HEAD_DIM] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5 for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                x_attn.extend(sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(HEAD_DIM))
            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
        return linear(x, state_dict['lm_head'])

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        keys, values = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            losses.append(-probs[target_id].log())
        loss = (1 / n) * sum(losses)
        loss.backward()
        lr_t = learning_rate * (1 - step / NUM_STEPS)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0
        print(f"step {step + 1:4d} / {NUM_STEPS} | loss {loss.data:.4f}", end='\r', flush=True)
    print()

    checkpoint = {
        'version': 2,
        'model_config': {
            'chars': uchars,
            'n_layer': N_LAYER,
            'n_embd': N_EMBD,
            'block_size': BLOCK_SIZE,
            'n_head': N_HEAD,
        },
        'target_steps': NUM_STEPS,
        'completed_steps': NUM_STEPS,
        'params': [p.data for p in params],
        'm': m,
        'v': v,
        'random_state': random.getstate(),
    }
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    tmp = CKPT_PATH + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(checkpoint, f)
    os.replace(tmp, CKPT_PATH)
    print(f"checkpoint saved: {CKPT_PATH}")
    return checkpoint


def build_state_dict(flat, vocab_size):
    pos = 0

    def take(nout, nin):
        nonlocal pos
        mat = [flat[pos + r * nin:pos + (r + 1) * nin] for r in range(nout)]
        pos += nout * nin
        return mat

    sd = {'wte': take(vocab_size, N_EMBD), 'wpe': take(BLOCK_SIZE, N_EMBD), 'lm_head': take(vocab_size, N_EMBD)}
    for i in range(N_LAYER):
        sd[f'layer{i}.attn_wq'] = take(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wk'] = take(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wv'] = take(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wo'] = take(N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc1'] = take(4 * N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc2'] = take(N_EMBD, 4 * N_EMBD)
    assert pos == len(flat), (pos, len(flat))
    return sd


lin = lambda x, w: [sum(a * b for a, b in zip(wo, x)) for wo in w]


def softmax_f(l):
    mx = max(l)
    e = [math.exp(v - mx) for v in l]
    s = sum(e)
    return [v / s for v in e]


def rms_f(x):
    ms = sum(v * v for v in x) / len(x)
    return [v * (ms + 1e-5) ** -0.5 for v in x]


def r4(x):
    if isinstance(x, list):
        return [r4(v) for v in x]
    return round(x, 4)


def forward_step(sd, tid, pid, keys, values, vocab_size):
    rec = {'token_id': tid, 'pos_id': pid}
    tok_emb = sd['wte'][tid]
    pos_emb = sd['wpe'][pid]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    rec['tok_emb'], rec['pos_emb'], rec['x_sum'] = tok_emb, pos_emb, x
    x = rms_f(x)
    rec['x_norm0'] = x
    li = 0
    res = x
    x = rms_f(x)
    rec['ln1'] = x
    q = lin(x, sd[f'layer{li}.attn_wq'])
    k = lin(x, sd[f'layer{li}.attn_wk'])
    v = lin(x, sd[f'layer{li}.attn_wv'])
    rec['q'], rec['k'], rec['v'] = q, k, v
    keys[li].append(k)
    values[li].append(v)
    x_attn = []
    rec['heads'] = []
    for h in range(N_HEAD):
        hs = h * HEAD_DIM
        q_h = q[hs:hs + HEAD_DIM]
        k_h = [ki[hs:hs + HEAD_DIM] for ki in keys[li]]
        v_h = [vi[hs:hs + HEAD_DIM] for vi in values[li]]
        al = [sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5 for t in range(len(k_h))]
        aw = softmax_f(al)
        head_out = [sum(aw[t] * v_h[t][j] for t in range(len(v_h))) for j in range(HEAD_DIM)]
        rec['heads'].append({'q_h': q_h, 'attn_logits': al, 'attn_weights': aw, 'head_out': head_out})
        x_attn.extend(head_out)
    rec['x_attn'] = x_attn
    x = lin(x_attn, sd[f'layer{li}.attn_wo'])
    rec['wo_out'] = x
    x = [a + b for a, b in zip(x, res)]
    rec['resid1'] = x
    res = x
    x = rms_f(x)
    rec['ln2'] = x
    x = lin(x, sd[f'layer{li}.mlp_fc1'])
    rec['fc1_pre'] = x
    x = [max(0.0, v_) for v_ in x]
    rec['fc1_relu'] = x
    x = lin(x, sd[f'layer{li}.mlp_fc2'])
    rec['fc2'] = x
    x = [a + b for a, b in zip(x, res)]
    rec['resid2'] = x
    logits = lin(x, sd['lm_head'])
    rec['logits'] = logits
    rec['probs'] = softmax_f(logits)
    rec['probs_t05'] = softmax_f([l / 0.5 for l in logits])
    return rec


def run_and_dump(ck):
    cfg = ck['model_config'] if ck.get('version') == 2 else ck['config']
    chars = cfg['chars']
    stoi = {c: i for i, c in enumerate(chars)}
    BOS = len(chars)
    vocab_size = len(chars) + 1
    sd = build_state_dict(ck['params'], vocab_size)

    context = 'em'
    toks = [BOS] + [stoi[c] for c in context]
    labels = ['<BOS>'] + list(context)
    keys, values = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
    steps = [forward_step(sd, t, i, keys, values, vocab_size) for i, t in enumerate(toks)]

    token_labels = chars + ['<END>']
    final = steps[-1]
    top = sorted(range(vocab_size), key=lambda i: -final['probs_t05'][i])[:8]
    flow = {
        'meta': {
            'context': context,
            'labels': labels,
            'chars': chars,
            'token_labels': token_labels,
            'bos': BOS,
            'checkpoint': os.path.basename(CKPT_PATH),
            'completed_steps': ck.get('completed_steps'),
            'n_embd': N_EMBD, 'n_head': N_HEAD, 'head_dim': HEAD_DIM,
            'temperature': 0.5,
        },
        'steps': [{k: r4(v) if k not in ('token_id', 'pos_id', 'heads') else v for k, v in s.items()} for s in steps],
        'kv_cache': [
            {
                'label': labels[i],
                'k': r4(steps[i]['k']),
                'v': r4(steps[i]['v']),
                'k_heads': [r4(steps[i]['k'][h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_HEAD)],
                'v_heads': [r4(steps[i]['v'][h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(N_HEAD)],
            } for i in range(len(steps))
        ],
        'top': [{'ch': token_labels[i], 'p': r4(final['probs_t05'][i]), 'p_raw': r4(final['probs'][i])} for i in top],
    }
    for s in flow['steps']:
        s['heads'] = [{k: r4(v) for k, v in h.items()} for h in s['heads']]

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w') as f:
        json.dump(flow, f, ensure_ascii=False, indent=1)
    print(f"wrote {JSON_PATH}")

    # human-readable verification printout (matches dump_attention.py style)
    m_step = steps[-1]
    print(f"\n=== step pos=2 token='m' ===")
    for h, hd in enumerate(m_step['heads']):
        print(f"H{h}  q = [{', '.join(f'{v:+.2f}' for v in hd['q_h'])}]")
        for t, (l, w) in enumerate(zip(hd['attn_logits'], hd['attn_weights'])):
            print(f"    {labels[t]:>5}  logit={l:+.3f}  w={w:.3f} {'█' * round(w * 40)}")
    s_probs = sum(m_step['probs'])
    print(f"probs sum = {s_probs:.6f}")
    ok = all(abs(m_step['resid2'][j] - (m_step['fc2'][j] + m_step['resid1'][j])) < 1e-9 for j in range(N_EMBD))
    print(f"resid2 == fc2 + resid1: {ok}")
    print("top-8 after 'em' (T=0.5):")
    for e in flow['top']:
        print(f"   {e['ch']:>5}  {e['p']:.4f}  {'█' * round(e['p'] * 60)}")
    return flow


def inject_into_html(flow):
    if not os.path.exists(HTML_PATH):
        print("flow.html not found yet — skipping inline injection")
        return
    html = open(HTML_PATH, encoding='utf-8').read()
    start_marker, end_marker = '/*__FLOW_DATA_START__*/', '/*__FLOW_DATA_END__*/'
    start = html.index(start_marker) + len(start_marker)
    end = html.index(end_marker)
    payload = '\nconst FLOW = ' + json.dumps(flow, ensure_ascii=False) + ';\n'
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html[:start] + payload + html[end:])
    print(f"injected FLOW data into {HTML_PATH}")


if __name__ == '__main__':
    ck = ensure_checkpoint()
    flow = run_and_dump(ck)
    inject_into_html(flow)
