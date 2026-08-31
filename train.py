import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from datasets import load_dataset
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from torchtune.modules import RotaryPositionalEmbeddings
from transformers import AutoTokenizer

import storage

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')
repo_id = "kalashnikov-dev/miniLM"


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")


vocab_size = len(tokenizer)
batch_size = 64
seq_len = 2048
d_model = 768
d_ff = int(d_model * 8/3) # 2048
d_head = 64
n_heads = 12 
n_kv_heads = 4
n_layers = 16
lr = 2e-3
accumulation_steps = 2
total_steps = 115000
total_micro_steps = total_steps * accumulation_steps
warmup_steps = 2000
decay_steps = int(total_steps * 0.20)
stable_end = total_steps - decay_steps 




class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.atn_norm = nn.RMSNorm(d_model, eps=1e-5) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html, eps for bfloat stability
        self.q = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.k = nn.Linear(d_model, d_head * n_kv_heads, bias = False)
        self.v = nn.Linear(d_model, d_head * n_kv_heads, bias = False)
        self.o = nn.Linear(d_model, d_model, bias = False)

        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5) 
        self.pre_ff_norm = nn.RMSNorm(d_model, eps=1e-5)

        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_ff, d_model, bias = False)
        self.w3 = nn.Linear(d_model, d_ff, bias = False)

        std = 0.02
        nn.init.normal_(self.q.weight, mean=0.0, std=std)
        nn.init.normal_(self.k.weight, mean=0.0, std=std)
        nn.init.normal_(self.v.weight, mean=0.0, std=std)
        nn.init.normal_(self.w1.weight, mean=0.0, std=std)
        nn.init.normal_(self.w3.weight, mean=0.0, std=std)

        scaled_std = std / math.sqrt(2 * n_layers) # for residual projections 
        nn.init.normal_(self.o.weight, mean=0.0, std=scaled_std)
        nn.init.normal_(self.w2.weight, mean=0.0, std=scaled_std)


    def forward(self, x, rope):
        b, s, _ = x.shape

        x_norm = self.atn_norm(x)

        Q = self.q_norm(self.q(x_norm).view(b, s, n_heads, d_head))
        K = self.k_norm(self.k(x_norm).view(b, s, n_kv_heads, d_head))
        V = self.v(x_norm).view(b, s, n_kv_heads, d_head)

        Q_rotated = rope(Q)
        K_rotated = rope(K)

        x_atn = F.scaled_dot_product_attention(
            Q_rotated.transpose(1, 2), K_rotated.transpose(1, 2), V.transpose(1, 2), is_causal = True, enable_gqa = True) #  https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html

        x_atn = self.o(x_atn.transpose(1, 2).reshape(b, s, d_head * n_heads))

        x = x + x_atn

        x_norm = self.pre_ff_norm(x)

        swiglu = self.w2(self.w1(x_norm) * F.silu(self.w3(x_norm)))

        x = x + swiglu
        return x



class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = seq_len+1) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope
        self.post_ff_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.ff = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])

        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)


    def forward(self, tokens):
        x = self.lm_head(tokens)

        for block in self.ff:
            #x = block(x, self.rope)
            x = checkpoint(block, x, self.rope, use_reentrant=False) # 15% tok/s gain
        x = self.post_ff_norm(x)

        return x




class StreamingDataset(IterableDataset):
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        buf = []
        batch = []
        for x in self.stream:
            tokens = tokenizer.encode(x["text"], add_special_tokens=False)
            buf.extend(tokens)
            buf.append(tokenizer.eos_token_id)
                
            while len(buf) >= seq_len + 1:
                chunk = buf[:seq_len + 1]
                buf = buf[seq_len:]
                batch.append(torch.tensor(chunk))

                if len(batch) == batch_size:
                    yield torch.stack(batch)
                    batch = []





def get_lr_multiplier(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < stable_end:
        return 1.0
    p = min((step - stable_end) / decay_steps, 1.0)
    return max(0.0, 1.0 - math.sqrt(p))


raw_model = Model().to(device)

decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW(
    [{'params': decay_params, 'weight_decay': 0.1},{'params': no_decay_params, 'weight_decay': 0.0}],
    lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True
)
scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
fused_ce = LigerFusedLinearCrossEntropyLoss()

resume_micro_step = 0
docs_to_skip = 0
ckpt = storage.load_checkpoint(device)
if ckpt:
    raw_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    resume_step = ckpt['step']
    resume_micro_step = resume_step * accumulation_steps

    tokens_per_step = batch_size * seq_len * accumulation_steps
    tokens_seen = resume_step * tokens_per_step
    AVG_TOKENS_PER_DOC = 1000 
    docs_to_skip = int(tokens_seen / AVG_TOKENS_PER_DOC)
    if docs_to_skip > 0:
        print(f"skipping {docs_to_skip} documents...")


    print(f"loaded checkpoint {resume_step}")


stream = load_dataset(
    "HuggingFaceFW/fineweb-edu", 
    name="sample-100BT", 
    split="train", 
    streaming=True
).skip(docs_to_skip).shuffle(seed=1337, buffer_size=10_000)


dataset = StreamingDataset(stream)
dataloader = DataLoader(dataset, batch_size=None, pin_memory=True) 

model = torch.compile(raw_model)


running_loss = 0.0
t0 = time.perf_counter()
for micro_step, batch in enumerate(dataloader, start=resume_micro_step):
    if micro_step >= total_micro_steps: 
        break
    
    batch = batch.to(device, non_blocking=True)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        
        pre_logits = model(batch[:, :-1])
        targets = batch[:, 1:]

        loss = fused_ce(raw_model.lm_head.weight, pre_logits.reshape(-1, d_model), targets.reshape(-1))
        loss = loss / accumulation_steps

    loss.backward()
    running_loss += loss.detach()

    if (micro_step + 1) % accumulation_steps == 0:
        grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step = (micro_step + 1) // accumulation_steps
        if step % 250 == 0:
            storage.save_checkpoint(raw_model, optimizer, scheduler, step, repo_id)
        
        #prints
        n_tokens = batch_size * seq_len
        dt = time.perf_counter() - t0
        step_loss = running_loss.item()
        print(f"step {step} loss={step_loss:.4f}  grad_norm={grad_norm.item():.2f}  {dt:.2f}s {n_tokens*accumulation_steps / dt:.0f} tok/s")
        running_loss = 0.0
        t0 = time.perf_counter()


storage.save_final_model(raw_model, repo_id)
storage.wait_for_uploads()

