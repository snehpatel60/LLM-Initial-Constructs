from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import numpy as np
import time

# ----------------------------------------------------------------------------------------------------
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # take in a token embedding x, output Qx, Kx, and Vx.

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size (number of simultaneous training examples), sequence length (number of tokens), dimension of embedding space (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        #
        #
        qkv = self.c_attn(x) # x is the token embeddings, qkv is the combined outputs of the Q, K, and V matrices we're training
        q, k, v = qkv.split(self.n_embed, dim=2) # now we extract q, k, and v from the combined output
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # transpose T and self.n_head because head encapsulates the particular sequence length
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # pytorch will treat batches and heads all as batches, and will apply the q-k marriage and everything to the 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # dot q and k, divide by sqrt of "number of tokens?", mask, softmax
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assembled all head outputs side by side
        y = self.c_proj(y) # combine head outputs according to yet another trained weights function
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')  # no need to use approximation anymore, but gpt2 used the tanh approximation for gelu
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # words to embeddings
            wpe = nn.Embedding(config.block_size, config.n_embed), # words to positionional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # number of attention heads
            ln_f = nn.LayerNorm(config.n_embed), # final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme for wte (768 embedding space parameters * 50257 unique token identifiers)
        # We want tokens -> embeddings to be similar for similar tokens, but we want similar tokens to have similar probabilities
        # Therefore, same wte that converts tokens -> embeddings also converts embeddings -> token probabilities
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # typically std = 1/sqrt(number of input params) = 1/sqrt(embedding params)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size() # batch size (number of simultaneous training examples), sequence length (number of tokens)
        #  ------------------------------
        # | batch 1: lasihgoshtoihsodihg |
        # | batch 2: lshoilbhso;eiht;oih |
        # | batch 3: obhsoeithoshlsidhlt |
        #  ------------------------------
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # ^ the sequence length, aka time, cannot be greater than the block size, which is the maximum context window

        pos = torch.arange(0, T, dtype=torch.long).to(idx.device) # shape (T)
        # device needs to be the same device as idx, we want to train with a gpu

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)

        x = tok_emb + pos_emb # remember that we add our position embeddings to every single batch individually

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # cross entropy does not like 3 dim, which is (B, T, vocab_size)
            # flatten it out to 2 dim with view, flatten targets to 1 dim of B * T

        return logits, loss  # these are a softmax away from becoming probabilities over the entire vocabulary
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard these k's that are masks / buffers

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ----------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load tokens from disk and store them in memory
        # use v_mage.txt for Sneh's fanfiction, input.txt for Shakespeare
        with open('v_mage.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y


# try to connect to a device
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
    device = "mps"

print(f"using device: {device}")

# device = "cpu"
#----

#_____________
# Get some data
# with open('input.txt', 'r') as f:
#     text = f.read()
# data = text[:1000]
# print(data[:100])

# enc = tiktoken.get_encoding('gpt2')
# # tokens = enc.encode("Hello, I'm a language model,")
# tokens = enc.encode(data)

# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)
# print(x)
# print(y)
#___________

train_loader = DataLoaderLite(B=16, T=512)

# model = GPT.from_pretrained('gpt2') # with pretrained weights
model = GPT(GPTConfig())
model.to(device)

#optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

for i in range(100000): # 10,000 epochs takes about 12 hours on a macbook m1 max, produces a very meh model, but the practice was good.
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    logits, loss = model(x, y)
    
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    t1 = time.time()
    dt = (t1-t0)*1000 # time difference in milliseconds
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms")

# TODO: Keep a local copy of the trained matrices so we don't have to redo the training each time.

model.eval()  # now in evaluation mode
num_return_sequences = 4  # batch size, will also return this many responses for any input prompt
max_length = 512  # size of context window

enc = tiktoken.get_encoding('gpt2')

def query_model(input):
    tokens = enc.encode(input)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    while x.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    logits, _ = model.forward(x) # B, T, vocab_size

                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)

                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)

                    # do top-k sampling of 50 (huggingface pipeline default) because we never sample tokens lower than the top 50 most likely, no weird tokens
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                    # select a token from the top-k probabilities
                    ix = torch.multinomial(topk_probs, 1) # (B, 1)

                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

                    # append to the sequence
                    x = torch.cat((x, xcol), dim=1)

    #print!
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("\n>", decoded)



#-----

if __name__ == "__main__":
    while True:
        prompt = input("Prompt> ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        query_model(prompt)

