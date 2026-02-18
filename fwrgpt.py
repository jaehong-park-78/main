#fwrgpt
import math
import random

# ----------------------------------------------------------------------
# 데이터 및 토크나이저
# ----------------------------------------------------------------------
random.seed(42)
names = [
    "emma","olivia","ava","isabella","sophia","charlotte","mia","amelia",
    "harper","evelyn","abigail","elizabeth","sofia","avery","ella","scarlett",
    "grace","chloe","victoria","riley","aria","lily","aubrey","zoey",
    "penelope","layla","luna","nora","mila","emily","hazel","madison",
    "ellie","nova","leah","zoe","violet","stella","aurora","hannah",
    "elijah","noah","liam","mason","james","ethan","logan","benjamin",
    "alexander","oliver","jacob","michael","daniel","henry","jackson",
    "sebastian","aiden","matthew","samuel","david","joseph","carter",
    "owen","wyatt","jack","luke","jayden","dylan","grayson","levi",
    "isaac","gabriel","anthony","thomas","leo","mateo","ezra","theo"
]
docs = names * 25
random.shuffle(docs)

chars = sorted(set(''.join(docs)))
vocab_size = len(chars) + 1
BOS = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

# ----------------------------------------------------------------------
# Autograd Engine
# ----------------------------------------------------------------------
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exp):
        out = Value(self.data ** exp, (self,))
        def _backward():
            self.grad += exp * (self.data ** (exp-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-10), (self,))
        def _backward():
            self.grad += (1 / (self.data + 1e-10)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)

# ----------------------------------------------------------------------
# 모델 초기화
# ----------------------------------------------------------------------
n_embd, n_head, block_size, n_layer = 48, 6, 12, 2
head_dim = n_embd // n_head

def init_mat(r, c, std=0.02):
    return [[Value(random.gauss(0, std)) for _ in range(c)] for _ in range(r)]

params = {
    'wte': init_mat(vocab_size, n_embd),
    'wpe': init_mat(block_size, n_embd),
    'lm_head': init_mat(vocab_size, n_embd)
}
for li in range(n_layer):
    for n in ['wq','wk','wv','wo']:
        params[f'{n}_{li}'] = init_mat(n_embd, n_embd)
    params[f'mlp_fc1_{li}'] = init_mat(4*n_embd, n_embd)
    params[f'mlp_fc2_{li}'] = init_mat(n_embd, 4*n_embd)

flat_params = [p for mat in params.values() for row in mat for p in row]

# ----------------------------------------------------------------------
# FWR 메트릭 (복원 & 간소화)
# ----------------------------------------------------------------------
def compute_fwr(loss_val, params_list):
    loss = loss_val + 1e-8
    f = 1.0 / loss
    param_norm = math.sqrt(sum(p.data**2 for p in params_list) / len(params_list) + 1e-10)
    w = 1.0 / (param_norm + 0.01)  # 클리핑 효과
    r = math.exp(-loss * 1.8)
    g = math.sqrt(sum(p.grad**2 for p in params_list) / len(params_list) + 1e-8)
    e = f * w * r * math.log1p(g * 30)
    return f"{f:.3f}", f"{w:.3f}", f"{r:.3f}", f"{g:.4f}", f"{e:.4f}"

# ----------------------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------------------
def linear(x, W):
    return [sum(xi * wi for xi, wi in zip(x, row)) for row in W]

def softmax(vals):
    m = max(v.data for v in vals)
    exps = [(v - m).exp() for v in vals]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi * ((ms + 1e-5) ** -0.5) for xi in x]

# ----------------------------------------------------------------------
# GPT Forward (약간 최적화)
# ----------------------------------------------------------------------
def gpt_forward(token_idx, pos_idx, past_keys, past_values):
    x = rmsnorm([a + b for a, b in zip(params['wte'][token_idx], params['wpe'][pos_idx])])
    for li in range(n_layer):
        xr = rmsnorm(x)
        q = linear(xr, params[f'wq_{li}'])
        k = linear(xr, params[f'wk_{li}'])
        v = linear(xr, params[f'wv_{li}'])
        past_keys[li].append(k)
        past_values[li].append(v)

        attn_out = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs+head_dim]
            scores = [sum(qh[j] * pk[hs+j] for j in range(head_dim)) / math.sqrt(head_dim)
                      for pk in past_keys[li]]
            weights = softmax(scores)
            head_out = [sum(w * pv[hs+j] for w, pv in zip(weights, past_values[li]))
                        for j in range(head_dim)]
            attn_out.extend(head_out)

        x = [a + b for a, b in zip(linear(attn_out, params[f'wo_{li}']), x)]

        xr = rmsnorm(x)
        h = [xi.relu() for xi in linear(xr, params[f'mlp_fc1_{li}'])]
        x = [a + b for a, b in zip(linear(h, params[f'mlp_fc2_{li}']), x)]

    return linear(rmsnorm(x), params['lm_head'])

# ----------------------------------------------------------------------
# 학습 루프 + FWR 출력
# ----------------------------------------------------------------------
lr = 0.018
m = [0.0] * len(flat_params)
v_adam = [0.0] * len(flat_params)
beta1, beta2 = 0.9, 0.95

print("FWR-GPT 학습 시작... 한 방울이 모여 큰 한 방울이 됩니다.")

for step in range(4000):   # 3000 → 4000으로 약간 늘림
    doc = random.choice(docs)
    tokens = [BOS] + [stoi[c] for c in doc] + [BOS]
    seq_len = min(block_size, len(tokens)-1)

    past_k = [[] for _ in range(n_layer)]
    past_v = [[] for _ in range(n_layer)]
    losses = []

    for pos in range(seq_len):
        logits = gpt_forward(tokens[pos], pos, past_k, past_v)
        probs = softmax(logits)
        losses.append(-probs[tokens[pos+1]].log())

    loss = sum(losses) / len(losses)

    for p in flat_params:
        p.grad = 0.0
    loss.backward()

    f, w, r, g, e = compute_fwr(loss.data, flat_params)

    lr_t = lr * (1 - step / 5000) ** 0.5
    for i, p in enumerate(flat_params):
        grad = p.grad + 1e-4 * p.data
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * grad**2
        mh = m[i] / (1 - beta1**(step + 1))
        vh = v_adam[i] / (1 - beta2**(step + 1))
        p.data -= lr_t * mh / (math.sqrt(vh) + 1e-8)

    if step % 250 == 0:
        print(f"[{step:4d}] loss={loss.data:5.3f} | FWR-E: {e} (F{w} R{r} G{g})")

# ----------------------------------------------------------------------
# 생성 (temperature 적용)
# ----------------------------------------------------------------------
print("\n--- FWR Merged Names ---")
for _ in range(20):
    pk = [[] for _ in range(n_layer)]
    pv = [[] for _ in range(n_layer)]
    token = BOS
    name = []

    for pos in range(block_size):
        logits = gpt_forward(token, pos, pk, pv)
        # temperature 0.9 적용
        scaled = [l * 0.9 for l in logits]
        probs = softmax(scaled)
        token = random.choices(range(vocab_size), [p.data for p in probs])[0]
        if token == BOS:
            break
        name.append(itos[token])

    print(''.join(name) if name else "(empty)")

print("\n완료.")
