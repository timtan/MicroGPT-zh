"""Dump what the learned position embeddings (wpe) look like after training.

Usage: python3 scripts/dump_position.py [checkpoint.pkl]

Reproduces the numbers in TALK.md 圖 3b. Reads the checkpoint directly and
slices `params` in the same order microgpt.py builds `state_dict`:
    wte (vocab x n_embd), wpe (block_size x n_embd), lm_head (vocab x n_embd), ...
"""
import math
import pickle
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/jinyong-2l-e16-h4-15k-seed42.pkl'
ck = pickle.load(open(path, 'rb'))
cfg = ck['config'] if ck.get('version') == 1 else ck['model_config']
n_embd, block_size = cfg['n_embd'], cfg['block_size']
vocab_size = len(cfg['chars']) + 1

flat = ck['params']
wte = [flat[i * n_embd:(i + 1) * n_embd] for i in range(vocab_size)]
off = vocab_size * n_embd
wpe = [flat[off + i * n_embd:off + (i + 1) * n_embd] for i in range(block_size)]

norm = lambda v: math.sqrt(sum(x * x for x in v))
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))

wte_mean = sum(norm(v) for v in wte) / vocab_size
init_norm = 0.08 * math.sqrt(n_embd)  # matrix() initialises with gauss(0, 0.08)

print(f"checkpoint: {path}")
print(f"n_embd={n_embd}  block_size={block_size}  vocab={vocab_size}  steps={ck['completed_steps']}\n")

print("=== 每個位置向量的 L2 長度 = sqrt(sum(x^2)) ===")
print(f"（初始化時全部約 {init_norm:.2f}；wte 平均長度 {wte_mean:.2f}）\n")
for i, v in enumerate(wpe):
    n = norm(v)
    print(f"  pos {i}: {n:5.2f}  {'#' * round(n * 8)}")

print("\n=== 位置向量兩兩的 cosine 相似度 ===\n")
print("        " + "".join(f"  pos{j} " for j in range(block_size)))
for i in range(block_size):
    print(f"  pos{i} " + "".join(f" {cos(wpe[i], wpe[j]):+.2f} " for j in range(block_size)))

print("\n=== 每個位置向量的原始數值（看訊號集中在哪幾個維度）===\n")
for i, v in enumerate(wpe):
    biggest = max(range(n_embd), key=lambda j: abs(v[j]))
    print(f"  pos {i}: [" + ", ".join(f"{x:+.2f}" for x in v) + "]")
    print(f"          最大絕對值在第 {biggest} 維 ({v[biggest]:+.2f})，"
          f"L2={norm(v):.2f}，L1={sum(abs(x) for x in v):.2f}")
