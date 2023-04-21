import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
class OFDMmodel(nn.Module):
    """This is a multilayer convolutional neural network for classification of modulation signal datasets
    Args:  
    num_class: the number of the class
    Return: 
    logits (batch_size,num_class)
    """
    def __init__(self,num_class=6):#num_class is 
        super(OFDMmodel,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            )
        self.layer3=nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

        )
        self.layer4=nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)

        )
        self.poolinglayer=nn.MaxPool2d((1,4),stride=1)
        self.fc=nn.Linear(384,num_class)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.poolinglayer(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x







class CausalSelfAttention(nn.Module):
    """
   This is a multi-layer Transformer model similar to GPT for for classification of modulation signal datasets
   Args:
   n_embd: embedding dimensionality(signal length)
   n_head: heads of the Transformer
   attn_drop: Dropout
   resid_drop: Dropout
   signal_size: signal width

    """

    def __init__(self,n_embd=1024,n_head=4,attn_drop = 0.1,resid_drop = 0.1,signal_size=6):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd,n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(signal_size, signal_size))
                                     .view(1, 1, signal_size, signal_size))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, signal width, embedding dimensionality (n_embd)

        # calculate query, key, values
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble the outputs

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block 
    Args:
    n_embd: embedding dimensionality(signal length)
    resid_drop: Dropout
    
    
    """

    def __init__(self, n_embd=1024,resid_drop = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """a small signal GPT
    Args:
    n_embd: embedding dimensionality(signal length)
    signal_size: signal width
    embd_drop: Dropout
    n_layer: the number of the layer
    num_class: the number of the class
    Return: 
    logits(it has the same dimensions as the input)
    
    """
    def __init__(self,signal_size=6,n_embd=1024,embd_drop=0.1,n_layer=5,classes=6):
        super(GPT,self).__init__()
        self.transformer = nn.ModuleDict(dict(
            
            wpe = nn.Embedding(signal_size, n_embd),
            drop = nn.Dropout(embd_drop),
            h = nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd*signal_size,classes, bias=False)
        self.padding=nn.ZeroPad2d(padding=(0,0,2,2))

    def forward(self,x):
        x=self.padding(x)
        b,t,n=x.size()
        pos = torch.arange(0, t, dtype=torch.long,device=torch.device("cuda")).unsqueeze(0)
        tok_emb = x
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x=x.view(x.size(0),-1)
        logits = self.lm_head(x)
        return logits













