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
tokens = torch.tensor(tokenizer.encode(training_data))
test_tokens = torch.tensor(tokenizer.encode(["Hello, I'm a language model and I'm going to generate a",
                 "Hello, I'm a language model and I'm going to code really"]))

#hyperparameters
vocab_size = tokenizer.vocab_size
batch_size = 2
seq_len = 15
context_len = 2048
d_model = 576
d_ff = int(d_model * 8/3)
d_head = 64
n_heads = 9 # d_model / d_head
n_layers = 10 # <-----

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.atn_norm = nn.RMSNorm(d_model) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
        self.q = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.k = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.v = nn.Linear(d_model, d_head * n_heads, bias = False)
        self.o = nn.Linear(d_model, d_model, bias = False)
        self.rope = RotaryPositionalEmbeddings(dim = d_head, max_seq_len = context_len) # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope
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
        self.post_ff_norm = nn.RMSNorm(d_model)
        self.ff = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
        

    def forward(self, tokens):
        x = self.lm_head(tokens)

        for block in self.ff:
            x = block(x)

        x = self.post_ff_norm(x)

        logits = x @ self.lm_head.weight.transpose(0,1)

        return logits


model = Model()
optimizer = torch.optim.AdamW(model.parameters())
epochs = 24

for _ in range(epochs):
    optimizer.zero_grad()

    logits = model(tokens)

    preds = logits[:, :-1, :]
    targets = tokens[:, 1:]
    loss = F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1)) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html

    loss.backward()
    optimizer.step()

    print(loss)


model.eval()
with torch.no_grad():
    test_logits = model(test_tokens)
    last_logits = test_logits[:, -1, :]
    probs = F.softmax(last_logits, dim=-1)

    idx_next = torch.multinomial(probs, num_samples=1)

    output_tokens = tokenizer.decode(idx_next)
    print(output_tokens)


