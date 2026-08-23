from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


#data
training_data = ["Hello, I'm a language model and I'm going to generate a text",
                 "Hello, I'm a language model and I'm going to code really nice"]

#tokenization
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokens = tokenizer.encode(training_data)

#hyperparameters
vocab_size = tokenizer.vocab_size
batch_size = 2
batch_len = 15
context_len = 2048
d_model = 576
d_head = 64
n_heads = 9 # d_model / d_head


#embedding
lm_head = torch.nn.Embedding(vocab_size, d_model)
x = lm_head(torch.tensor(tokens))

#attention
q = torch.nn.Linear(d_model, d_head * n_heads)
k = torch.nn.Linear(d_model, d_head * n_heads)
v = torch.nn.Linear(d_model, d_head * n_heads)

Q = q(x).view(batch_size, batch_len, n_heads, d_head)
K = k(x).view(batch_size, batch_len, n_heads, d_head)
V = v(x).view(batch_size, batch_len, n_heads, d_head)
print(Q.shape)
rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = context_len) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope

Q_rotated = rope(Q)
K_rotated = rope(K)

Q_rotated = Q_rotated.transpose(1, 2)
K_rotated = K_rotated.transpose(1, 2)
V = V.transpose(1, 2)
print(Q_rotated.shape)

x = F.scaled_dot_product_attention(Q_rotated, K_rotated, V, is_causal = True) #  https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
#enable_gqa = True


print(x.shape)


