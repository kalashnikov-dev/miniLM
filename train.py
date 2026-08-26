from datasets.iterable_dataset import StepExamplesIterable
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import math
from datasets import load_dataset
import time
from torch.optim.lr_scheduler import LambdaLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('high')



#tokenization
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
test_tokens = torch.tensor(tokenizer.encode(["Hello, I'm a language model and I'm going to generate a",
                 "Hello, I'm a language model and I'm going to code really"])).to(device)

#hyperparameters
vocab_size = len(tokenizer)
batch_size = 1
seq_len = 2048
context_len = 2048
d_model = 576
d_ff = int(d_model * 8/3)
d_head = 64
n_heads = 9 # d_model / d_head
n_layers = 30

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.atn_norm = nn.RMSNorm(d_model) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
        self.q = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.k = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.v = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.o = nn.Linear(d_model, d_model, bias = False)
        self.rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = context_len+1) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope
        self.pre_ff_norm = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_ff, d_model, bias = False)
        self.w3 = nn.Linear(d_model, d_ff, bias = False)

    def forward(self, x):
        b, s, _ = x.shape

        x_norm = self.atn_norm(x)

        Q = self.q(x_norm).view(b, s, n_heads, d_head)
        K = self.k(x_norm).view(b, s, n_heads, d_head)
        V = self.v(x_norm).view(b, s, n_heads, d_head)

        Q_rotated = self.rope(Q)
        K_rotated = self.rope(K)

        x_atn = F.scaled_dot_product_attention(
            Q_rotated.transpose(1, 2), K_rotated.transpose(1, 2), V.transpose(1, 2), is_causal = True) #  https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
        #enable_gqa = True

        x_atn = self.o(x_atn.transpose(1, 2).view(b, s, d_head * n_heads))

        x = x + x_atn

        x_norm = self.pre_ff_norm(x)

        swiglu = self.w2(self.w1(x_norm) * F.silu(self.w3(x_norm)))

        x = x + swiglu
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=(1/math.sqrt(d_model)))
        self.post_ff_norm = nn.RMSNorm(d_model)
        self.ff = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
        

    def forward(self, tokens):
        x = self.lm_head(tokens)

        for block in self.ff:
            x = block(x)

        x = self.post_ff_norm(x)

        logits = x @ self.lm_head.weight.transpose(0,1)

        return logits


total_steps = 1000
warmup_steps = total_steps * 0.1
decay_steps = total_steps * 0.2 

def get_lr_multiplier(step):
    if step < warmup_steps:
        return (step+1) / warmup_steps #first stop is nonzero due to +1
    
    elif step < total_steps - decay_steps:
        return 1.0

    else: 
        decay_step = step - (total_steps - decay_steps)
        return 1.0 - (decay_step / decay_steps)


model = Model()
model.to(device)    
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), fused = True)
stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
steps = 50

def get_batch(stream):
    buf = []
    batch = []

    for x in stream:
        tokens = tokenizer.encode(x["text"])
        buf.extend(tokens)
        buf.append(tokenizer.eos_token_id)
        
        while(len(buf) >= seq_len+1):
            chunk = buf[:seq_len+1]
            buf = buf[seq_len:]
            batch.append(torch.tensor(chunk))

            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
    

t0 = time.perf_counter()
for step, batch in enumerate(get_batch(stream)):
    if step >= steps: 
        break
    
    batch = batch.to(device)
    optimizer.zero_grad()
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(batch)

        preds = logits[:, :-1, :]
        targets = batch[:, 1:]
        loss = F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1)) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html

    loss.backward()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    n_tokens = batch_size * seq_len
    dt = time.perf_counter() - t0
    print(f"{step} loss={loss.item():.4f}  {dt:.2f}s {n_tokens / dt:.0f} tok/s")
    t0 = time.perf_counter()


model.eval()
with torch.no_grad():

    for i in range(10):
        test_logits = model(test_tokens)
        last_logits = test_logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        output_tokens = tokenizer.decode(idx_next)
        
        test_tokens = torch.cat([test_tokens, idx_next], dim=1)


    print(tokenizer.decode(test_tokens))

#TO DO
#grad clipping
#weight_decay
#exact lr (decide if grad accumulation is needed + ddp/FSDP)
#gqa