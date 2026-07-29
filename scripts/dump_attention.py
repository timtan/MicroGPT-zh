"""Dump attention weights for a prefix.

usage: python3 scripts/dump_attention.py [prefix] [checkpoint.pkl]
"""
import pickle, math, sys
ckpt_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/jinyong-2l-e16-h4-15k-seed42.pkl'
ck = pickle.load(open(ckpt_path,'rb'))
cfg = ck['config']; chars = cfg['chars']
n_layer, n_embd, n_head, block_size = cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['block_size']
head_dim = n_embd // n_head
stoi = {c:i for i,c in enumerate(chars)}; itos={i:c for i,c in enumerate(chars)}
BOS = len(chars); vocab_size = len(chars)+1
flat = ck['params']; pos = 0
def take(nout, nin):
    global pos
    m = [flat[pos+r*nin:pos+(r+1)*nin] for r in range(nout)]
    pos += nout*nin
    return m
sd = {}
sd['wte']=take(vocab_size,n_embd); sd['wpe']=take(block_size,n_embd); sd['lm_head']=take(vocab_size,n_embd)
for i in range(n_layer):
    sd[f'layer{i}.attn_wq']=take(n_embd,n_embd); sd[f'layer{i}.attn_wk']=take(n_embd,n_embd)
    sd[f'layer{i}.attn_wv']=take(n_embd,n_embd); sd[f'layer{i}.attn_wo']=take(n_embd,n_embd)
    sd[f'layer{i}.mlp_fc1']=take(4*n_embd,n_embd); sd[f'layer{i}.mlp_fc2']=take(n_embd,4*n_embd)
assert pos == len(flat), (pos, len(flat))
lin = lambda x,w: [sum(a*b for a,b in zip(wo,x)) for wo in w]
def softmax(l):
    mx=max(l); e=[math.exp(v-mx) for v in l]; s=sum(e); return [v/s for v in e]
def rms(x):
    ms=sum(v*v for v in x)/len(x); sc=(ms+1e-5)**-0.5; return [v*sc for v in x]
DUMP=[]
def gpt(tid,pid,keys,values):
    x=[t+p for t,p in zip(sd['wte'][tid], sd['wpe'][pid])]
    x=rms(x)
    for li in range(n_layer):
        res=x; x=rms(x)
        q=lin(x,sd[f'layer{li}.attn_wq']); k=lin(x,sd[f'layer{li}.attn_wk']); v=lin(x,sd[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        xa=[]
        for h in range(n_head):
            hs=h*head_dim
            qh=q[hs:hs+head_dim]; kh=[ki[hs:hs+head_dim] for ki in keys[li]]; vh=[vi[hs:hs+head_dim] for vi in values[li]]
            al=[sum(qh[j]*kh[t][j] for j in range(head_dim))/head_dim**0.5 for t in range(len(kh))]
            aw=softmax(al)
            DUMP.append((pid,li,h,qh,[kh[t] for t in range(len(kh))],al,aw))
            xa.extend([sum(aw[t]*vh[t][j] for t in range(len(vh))) for j in range(head_dim)])
        x=lin(xa,sd[f'layer{li}.attn_wo']); x=[a+b for a,b in zip(x,res)]
        res=x; x=rms(x); x=lin(x,sd[f'layer{li}.mlp_fc1']); x=[max(0.0,v) for v in x]
        x=lin(x,sd[f'layer{li}.mlp_fc2']); x=[a+b for a,b in zip(x,res)]
    return lin(x,sd['lm_head'])

name = sys.argv[1] if len(sys.argv)>1 else '歐陽克'
toks=[BOS]+[stoi[c] for c in name]
labels=['BOS']+list(name)
keys=[[] for _ in range(n_layer)]; values=[[] for _ in range(n_layer)]
for i,t in enumerate(toks):
    logits=gpt(t,i,keys,values)
for pid,li,h,qh,khs,al,aw in DUMP:
    if pid != len(toks)-1: continue
    print(f"L{li} H{h}  pos={pid} (查詢字='{labels[pid]}')")
    print(f"   q = [{', '.join(f'{v:+.2f}' for v in qh)}]")
    for t,(kv,l,w) in enumerate(zip(khs,al,aw)):
        bar='█'*round(w*40)
        print(f"   k({labels[t]:>3}) = [{', '.join(f'{v:+.2f}' for v in kv)}]  logit={l:+.2f}  w={w:.3f} {bar}")
    print()
probs=softmax(logits)
top=sorted(range(vocab_size), key=lambda i:-probs[i])[:8]
print("下一個 token 機率 top-8:")
for i in top:
    print(f"   {'<END>' if i==BOS else itos[i]}  {probs[i]:.4f}  {'█'*round(probs[i]*60)}")
