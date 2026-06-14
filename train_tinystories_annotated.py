## 0. Dataset & DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

data_path = Path("data/tinystories.txt")

text = open(data_path, "r", encoding="utf-8").read()

text = open("data/shakespeare.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))                                      # 텍스트에 등장하는 모든 문자들을 중복 없이 모아서 정렬
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)                                              # TinyStories에서는 약 90개 문자(알파벳, 숫자, 문장부호 등)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size                      # 끝에서 block size를 제외한 부분까지만 가능

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]                   # idx부터 idx + self.block_size까지
        y = self.data[idx + 1 : idx + self.block_size + 1]           # idx+1부터 idx + self.block_size + 1까지 (x보다 한칸씩 밀림)
        return x, y

block_size = 64
dataset = NextTokenDataset(data, block_size)                         # 전체 텍스트 데이터를 이용해서 x, y를 생성할 수 있는 dataset을 만듦
loader = DataLoader(dataset, batch_size=64, shuffle=True)            # dataloader가 dataset에서 샘플들을 꺼내서 학습시킴
xb, yb = next(iter(loader))                                          # 첫번째 batch를 꺼내서 확인하는 용도


## 1. Multi-head Attention
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)         # 입력dim=emb_dim, 출력dim=head_size
        self.query = nn.Linear(emb_dim, head_size, bias=False)       # "
        self.value = nn.Linear(emb_dim, head_size, bias=False)       # "
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix 생성
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape          #batch size(한 번에 몇개의 문장을 학습하는가), time step(각 문장이 몇글자인지), channel(embedding dimension)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # lower triangular matrix의 0 부분에 -inf를 채워넣어서 미래를 볼 수 없도록
        wei = F.softmax(wei, dim=-1)                                    # attention weight를 확률값처럼 만듦
        wei = self.dropout(wei)                                         # 일부 연결을 랜덤하게 끊어서 overfitting 방지
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):                                    # 여러 head를 묶음
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):    # dropout: 90%는 유지하고 10%만 제거
        super().__init__()                                              # 부모 class(=nn.Module)의 init을 실행하라
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])    # head 여러개 생성
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)             # 각 head를 출력해서 이어붙이면 (64, 64, 32)->(64, 64, 128)
        out = self.proj(out)                                            # shape은 유지되지만, 이어붙이기만 하면 정보가 통합이 안되므로 linear projection 다시해서 정보 통합
        out = self.dropout(out)                                         # overfitting 방지
        return out                                                      # 최종 shape = (B,T,C) = (64,64,128)
    

## 2. FeedForward & Block
class FeedForward(nn.Module):                                           # ff: attention이 모아온 정보(x)를 각 토큰별로 복잡하게 가공
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),                            # 128차원->512차원으로 보내서 더 복잡한 패턴을 학습하게 함
            nn.ReLU(),                                                  # 비선형성을 추가해서 신경망이 단순한 선형변환만 하지 않게 함 -> 복잡한 패턴 학습가능
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):                                               # forward: 입력이 들어왔을 때 어떤 계산을 할지 정의하는 함수
        return self.net(x)                                              # 입력이 들어오면 linear->relu->linear->dropout을 한번에 실행

class Block(nn.Module):                                                 # transformer block 하나
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)                                        # 입력 벡터를 normalize (너무 크거나 작지 않게), shape는 불변
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)   # self attention layer
        self.ln2 = nn.LayerNorm(emb_dim)                                        # feedforward하기 전에 한번 더 normalize
        self.ffwd = FeedForward(emb_dim, dropout)                               # feedforward layer

    def forward(self, x):
        x = x + self.sa(self.ln1(x))                                   # 원래 입력 더하기 for 원본 정보 보존 (residual connection)
        x = x + self.ffwd(self.ln2(x))
        return x


## 3. Tiny GPT
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1): # num_layers=4 -> block 네개 통과
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)                         # 숫자로 바뀐 토큰을 128차원 벡터로 바꿔줌 (64,64)->(64,64,128)
        self.position_embedding = nn.Embedding(block_size, emb_dim)                      # 같은 a이더라도 위치에 따라 의미가 다르므로 위치정보 추가 (64,128)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)                  # 최종적으로 normalize
        self.lm_head = nn.Linear(emb_dim, vocab_size)      # 128차원 특징을 90개의 문자 중 하나로 바꿈

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)             # 위치 생성
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]           # tok와 pos의 shape을 맞춰줘야 하므로.(64,128)->(1,64,128) (NONE은 unsqueeze(0)과 같은 의미)
        h = tok + pos                                      # 무슨 문자인지 & 어떤 위치인지 (64,64,128). pos의 (1,64,128)을 자동으로 (64,64,128)로 broadcasting해서 계산
        h = self.blocks(h)                                 # blocks 통과하면: 입력->b1->b2->b3->b4->출력
        h = self.ln_f(h)
        logits = self.lm_head(h)                           # (B,T,128)->(B,T,90) shape 바뀜
        return logits                                      # 다음 문자가 뭘지 예측하는 결과

model = TinyGPT(vocab_size, block_size)
logits = model(xb)                                         # TinyGPT.forward() 호출 -> logits[b,t,c]: b번째 문장의 t번째 위치에서 c가 나올 가능성
print("logits.shape:", logits.shape)


## 4. 학습
def sequence_cross_entropy(logits, targets):                                    # cross_entropy: 모델이 정답 문자를 얼마나 잘 맞췄는지를 측정하는 함수
    return F.cross_entropy(logits.transpose(1, 2), targets)                     # transpose: (64,64,90)->(64,90,64)
def train_one_epoch(model, loader, optimizer, device, max_steps=None):          # 한 epoch동안 학습하는 함수
    model.train()                                                               # model을 학습 모드로
    total_loss, total_count = 0.0, 0                                            # 나중에 평균 loss 계산용으로 미리 만들어놓음
    for step, (xb, yb) in enumerate(loader):                                    # 매 step마다 새로운 batch를 가져옴
        xb, yb = xb.to(device), yb.to(device)                                   # GPU 사용 가능하면 CPU에서 GPU로 이동
        logits = model(xb)                                                      # 모델 실행 (64,64,90)
        loss = sequence_cross_entropy(logits, yb)                               # loss 계산, 결과는 tensor() 형태
        optimizer.zero_grad()                                                   # Pytorch는 원래 gradient를 누적하므로 매 step마다 초기화 필요
        loss.backward()                                                         # back propagation (뒤에서부터 gradient 계산)
        optimizer.step()                                                        # weight를 update
        total_loss += loss.item() * xb.size(0)                                  # loss 누적
        total_count += xb.size(0)                                               # sample 수 누적 -> 현재 batch 수 더해짐
        if max_steps is not None and step + 1 >= max_steps:                     # TinyStories는 데이터가 더 크므로 학습시간을 줄이기 위해 100번째 batch까지만 학습
            break
    return total_loss / total_count                                             # 평균 loss 계산

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(vocab_size, block_size).to(device)                              # 모델 자체도 GPU에 올려야 함
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)                      # model.parameters(): model 안의 전체 학습 가능한 파라미터, lr: learning rate=0.0003

for epoch in range(100):                                                        # 100번 반복학습
    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=100) # 한 epoch 학습 후 평균 loss 반환
    print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")


## 5. Sampling
@torch.no_grad()                                                                # 생성할 땐 gradient 계산 안함 (학습할 때만 함)
def sample_gpt(model, block_size, stoi, itos, device, start_text="Once upon a time", max_new_tokens=400): # 최대 400글자 생성
    model.eval()                                                                # 평가 모드 -> Dropout(0.1) 비활성화
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)    # (0,0,0,,,,,0) shape (1,64)
    for ch in start_text:                                                       # start text 넣기
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)                    # 맨 앞은 제거, 맨 뒤에는 추가
    out = list(start_text)                                                      # 출력 문자열 저장
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]                                               # 우리는 다음 글자 하나만 필요하니까 마지막 위치만 사용
        probs = F.softmax(logits, dim=-1)                                       # 확률로 변환
        ix = torch.multinomial(probs, num_samples=1)                            # 문자 샘플링 (다양하게 뽑혀야 하므로 argmax 쓰면 너무 단조로워짐)
        out.append(itos[ix.item()])                                             # 출력에 GPT가 예상한 다음 글자 붙이기
        context = torch.cat([context[:, 1:], ix], dim=1)                        # 예상한 다음 글자까지로 context 업데이트
    return "".join(out)                                                         # 반환할 때 글자 배열 읽기 쉽도록 바꾸기

print(sample_gpt(model, block_size, stoi, itos, device, start_text="Once upon a time", max_new_tokens=500))