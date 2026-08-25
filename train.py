from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


#data
training_data = ["Hello, I'm a language model and I'm going to generate a text",
                 "Hello, I'm a language model and I'm going to code really good"]
#tokenization
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokens = tokenizer.encode(training_data)
tokens = torch.tensor(tokens)

#hyperparameters
vocab_size = tokenizer.vocab_size
batch_size = 2
seq_len = 15
context_len = 2048
d_model = 576
d_ff = int(d_model * 8/3)
d_head = 64
n_heads = 9 # d_model / d_head



#embedding
lm_head = nn.Embedding(vocab_size, d_model)
x = lm_head(tokens)

#pre attention normalization
atn_norm = nn.RMSNorm(d_model) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
x_norm = atn_norm(x)


#attention
q = nn.Linear(d_model, d_head * n_heads, bias = False)
k = nn.Linear(d_model, d_head * n_heads, bias = False)
v = nn.Linear(d_model, d_head * n_heads, bias = False)
o = nn.Linear(d_model, d_model, bias = False)

Q = q(x_norm).view(batch_size, seq_len, n_heads, d_head)
K = k(x_norm).view(batch_size, seq_len, n_heads, d_head)
V = v(x_norm).view(batch_size, seq_len, n_heads, d_head)

#positional encoding
rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = context_len) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope

Q_rotated = rope(Q)
K_rotated = rope(K)

x_atn = F.scaled_dot_product_attention(
    Q_rotated.transpose(1, 2), K_rotated.transpose(1, 2), V.transpose(1, 2), is_causal = True) #  https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
#enable_gqa = True

x_atn = x_atn.transpose(1, 2).view(batch_size, seq_len, d_head * n_heads)
x_atn = o(x_atn)

#residual conn
x = x + x_atn

#pre-FF normalization
pre_ff_norm = nn.RMSNorm(d_model)
x_norm = pre_ff_norm(x)

w1 = nn.Linear(d_model, d_ff, bias = False)
w2 = nn.Linear(d_ff, d_model, bias = False)
w3 = nn.Linear(d_model, d_ff, bias = False)

swiglu = w2(w1(x_norm) * F.silu(w3(x_norm)))

x = x + swiglu


#post-FF normalization
post_ff_norm = nn.RMSNorm(d_model)
x = post_ff_norm(x)

#output
logits = x @ lm_head.weight.transpose(0,1)

#loss
preds = logits[:, :-1, :]
targets = tokens[:, 1:]

loss = F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1)) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html
print(loss)

#backward pass
loss.backward()





