from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import math
from datasets import load_dataset
import time
from torch.optim.lr_scheduler import LambdaLR
import storage


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")


vocab_size = len(tokenizer)
batch_size = 1
seq_len = 2048
d_model = 576
d_ff = int(d_model * 8/3)
d_head = 64
n_heads = 9 # d_model / d_head
n_layers = 30
accumulation_steps = 128
total_steps = 100
total_micro_steps = total_steps * accumulation_steps
warmup_steps = int(total_steps * 0.1)
decay_steps = int(total_steps * 0.2)
repo_id = "kalashnikov-dev/miniLM"



class TransformerBlock(nn.Module):


    def __init__(self):
        super().__init__()
        self.atn_norm = nn.RMSNorm(d_model) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
        self.q = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.k = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.v = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.o = nn.Linear(d_model, d_model, bias = False)
        self.pre_ff_norm = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_ff, d_model, bias = False)
        self.w3 = nn.Linear(d_model, d_ff, bias = False)


    def forward(self, x, rope):
        b, s, _ = x.shape

        x_norm = self.atn_norm(x)

        Q = self.q(x_norm).view(b, s, n_heads, d_head)
        K = self.k(x_norm).view(b, s, n_heads, d_head)
        V = self.v(x_norm).view(b, s, n_heads, d_head)

        Q_rotated = rope(Q)
        K_rotated = rope(K)

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
        self.rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = seq_len+1) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope
        self.post_ff_norm = nn.RMSNorm(d_model)
        self.ff = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])


    def forward(self, tokens):
        x = self.lm_head(tokens)

        for block in self.ff:
            x = block(x, self.rope)

        x = self.post_ff_norm(x)

        logits = x @ self.lm_head.weight.transpose(0,1)

        return logits



def get_lr_multiplier(step):
    if step < warmup_steps:
        return (step+1) / warmup_steps # first stop is nonzero due to +1
    
    elif step < total_steps - decay_steps:
        return 1.0

    else: 
        decay_step = step - (total_steps - decay_steps)
        return 1.0 - (decay_step / decay_steps)


raw_model = Model().to(device)   
optimizer = torch.optim.AdamW(raw_model.parameters(), fused = True)
scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
model = torch.compile(raw_model)

stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)


def get_batch(stream):
    buf = []
    batch = []

    for x in stream:
        tokens = tokenizer.encode(x["text"], add_special_tokens=False)
        buf.extend(tokens)
        buf.append(tokenizer.eos_token_id)
        
        while len(buf) >= seq_len+1:
            chunk = buf[:seq_len+1]
            buf = buf[seq_len:]
            batch.append(torch.tensor(chunk))

            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
    


t0 = time.perf_counter()
for micro_step, batch in enumerate(get_batch(stream)):
    if micro_step >= total_micro_steps: 
        break
    
    batch = batch.to(device)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html
        loss = loss / accumulation_steps

    loss.backward()

    if (micro_step + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step = (micro_step + 1) // accumulation_steps
        if step == total_steps: # step % 1000 == 0
            storage.save_checkpoint(raw_model, optimizer, scheduler, step, repo_id)
        
        #prints
        torch.cuda.synchronize()
        n_tokens = batch_size * seq_len
        dt = time.perf_counter() - t0
        print(f"step {step} loss={loss.item()*accumulation_steps:.4f}  {dt:.2f}s {n_tokens*accumulation_steps / dt:.0f} tok/s")
        t0 = time.perf_counter()



storage.save_final_model(raw_model, repo_id)
storage.wait_for_uploads()


#sampling
test_tokens = torch.tensor(tokenizer.encode(["Hello, I'm a language model and"])).to(device)
model.eval()
with torch.no_grad():

    for _ in range(5):

        buffer = test_tokens

        for _ in range(10):
            test_logits = model(buffer)
            last_logits = test_logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            output_tokens = tokenizer.decode(idx_next)
            
            buffer = torch.cat([buffer, idx_next], dim=1)

        print(tokenizer.decode(buffer[0]))





#TO DO
#exact weight_decay + exact lr
#ddp/FSDP if needed
#gqa
#loading weights