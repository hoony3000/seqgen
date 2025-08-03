"""seq_sim_transformer.py

이 모듈은 NAND 시퀀스 생성 문제를 Transformer 기반 아키텍처로 해결하는 예시를 제공한다.
슬라이딩 윈도우 방식의 데이터셋을 사용하여 과거 `lookback`개의 이벤트로부터 다음 이벤트의
명령과 주소를 예측한다. 입력 피처에는 명령 ID, 주소 벡터와 함께 두 가지 추가 피처
(`delta_program_page`, `delta_read_page`)가 포함된다.
"""

import os
import pathlib
import tempfile
import types
import sys
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 안전한 임시 디렉터리 설정 (seq_sim_baseline과 동일)
# [setup 코드 생략 – 실제 파일에서는 TMPDIR 설정과 torch._dynamo 패치가 포함됨]

try:
    from seq_sim import ContrastiveDataGenerator, MAX_ADDR, CMD_VOCAB
except ImportError:
    raise ImportError(
        "seq_sim.py must be present in the same directory or PYTHONPATH to "
        "import ContrastiveDataGenerator and constants."
    )

MAX_PAGE = MAX_ADDR['page']

def compute_feature_sequence(seq: List[Tuple[int, List[float]]]) -> List[List[float]]:
    """Compute augmented feature vectors for each event in the sequence."""
    feature_seq: List[List[float]] = []
    last_program_page: dict = {}
    for cmd_id, addr_vec in seq:
        key = tuple(round(addr_vec[i] * MAX_ADDR[k]) for i, k in enumerate(['die','plane','block']))
        page = round(addr_vec[3] * MAX_PAGE)
        if cmd_id == CMD_VOCAB['program']:
            last_page = last_program_page.get(key, -1)
            delta_prog = (page - last_page) / float(MAX_PAGE) if last_page >= 0 else 0.0
        else:
            delta_prog = 0.0
        if cmd_id == CMD_VOCAB['read']:
            last_page = last_program_page.get(key, -1)
            delta_read = (last_page - page) / float(MAX_PAGE) if last_page >= 0 else 0.0
        else:
            delta_read = 0.0
        if cmd_id == CMD_VOCAB['program']:
            last_program_page[key] = page
        cmd_norm = cmd_id / float(len(CMD_VOCAB) - 1) if len(CMD_VOCAB) > 1 else 0.0
        feature_seq.append([cmd_norm] + addr_vec + [delta_prog, delta_read])
    return feature_seq

class GenerativeDataset(Dataset):
    """Dataset producing sliding-window samples for next-event prediction."""
    def __init__(self, sequences: List[List[Tuple[int, List[float]]]], lookback: int = 10):
        self.samples: List[Tuple[np.ndarray,int,np.ndarray]] = []
        for seq in sequences:
            if len(seq) <= lookback:
                continue
            features = compute_feature_sequence(seq)
            for t in range(lookback, len(seq)):
                window = np.array(features[t - lookback : t], dtype=np.float32)
                cmd_label, addr_label = seq[t]
                addr_arr = np.array(addr_label, dtype=np.float32)
                self.samples.append((window, cmd_label, addr_arr))
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int):
        return self.samples[idx]

class GRUModel(nn.Module):
    """GRU-based sequence model for next-step prediction."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.cmd_head = nn.Linear(hidden_dim, len(CMD_VOCAB))
        self.addr_head = nn.Linear(hidden_dim, 4)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_out, h_n = self.gru(x)
        last_h = h_n[-1]
        cmd_logits = self.cmd_head(last_h)
        addr_pred = torch.sigmoid(self.addr_head(last_h))
        return cmd_logits, addr_pred

class TransformerModel(nn.Module):
    """Transformer encoder model for next-event prediction."""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, max_seq_len: int = 50):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cmd_head = nn.Linear(d_model, len(CMD_VOCAB))
        self.addr_head = nn.Linear(d_model, 4)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        x_emb = self.input_fc(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(pos_ids)
        src = (x_emb + pos_emb).transpose(0, 1)  # (seq_len, batch, d_model)
        encoded = self.encoder(src)
        last_output = encoded[-1]
        cmd_logits = self.cmd_head(last_output)
        addr_pred = torch.sigmoid(self.addr_head(last_output))
        return cmd_logits, addr_pred

class PointerNetwork(nn.Module):
    """입력 시퀀스에 어텐션을 적용하여 다음 주소를 예측하는 포인터 네트워크."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        # 인코더: GRU를 사용하여 입력 시퀀스를 인코딩
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 어텐션 메커니즘: 포인터 가중치를 계산
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
        
        # 커맨드 예측을 위한 헤드
        self.cmd_head = nn.Linear(hidden_dim, len(CMD_VOCAB))

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, input_dim)
        enc_out, h_n = self.encoder(x)
        # enc_out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)

        last_h = h_n[-1]  # 마지막 은닉 상태를 어텐션 쿼리로 사용

        # Bahdanau-style 어텐션
        # W1(enc_out) shape: (batch_size, seq_len, hidden_dim)
        # W2(last_h).unsqueeze(1) shape: (batch_size, 1, hidden_dim)
        # 덧셈은 시퀀스 길이에 걸쳐 브로드캐스팅됨
        scores = self.V(torch.tanh(self.W1(enc_out) + self.W2(last_h).unsqueeze(1)))
        scores = scores.squeeze(2)  # Shape: (batch_size, seq_len)

        # 포인터는 어텐션 점수의 소프트맥스 값
        addr_pointer = F.softmax(scores, dim=1)

        # 마지막 은닉 상태로부터 커맨드를 예측
        cmd_logits = self.cmd_head(last_h)

        # 입력 주소들의 가중 평균으로 주소를 예측
        # 입력 피처의 1~4번 인덱스가 주소에 해당
        input_addresses = x[:, :, 1:5]  # Shape: (batch_size, seq_len, 4)
        
        # 포인터를 사용하여 입력 주소들의 가중치를 계산
        # addr_pointer.unsqueeze(1) shape: (batch_size, 1, seq_len)
        # bmm 결과 shape: (batch_size, 1, 4)
        addr_pred = torch.bmm(addr_pointer.unsqueeze(1), input_addresses).squeeze(1)
        # 최종 addr_pred shape: (batch_size, 4)

        return cmd_logits, addr_pred

# train(), evaluate(), prepare_sequences(), main() 함수는 GRU 버전과 동일한 구조입니다.
def train(model: nn.Module, dataloader: DataLoader, device: torch.device, num_epochs: int = 5, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cmd_loss_fn = nn.CrossEntropyLoss()
    addr_loss_fn = nn.SmoothL1Loss()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_cmd_loss = 0.0
        total_addr_loss = 0.0
        for x_batch, cmd_label, addr_label in dataloader:
            x_batch = x_batch.to(device)
            cmd_label = cmd_label.to(device)
            addr_label = addr_label.to(device)
            optimizer.zero_grad()
            cmd_logits, addr_pred = model(x_batch)
            cmd_loss = cmd_loss_fn(cmd_logits, cmd_label)
            addr_loss = addr_loss_fn(addr_pred, addr_label)
            loss = cmd_loss + addr_loss
            loss.backward()
            optimizer.step()
            total_cmd_loss += cmd_loss.item() * x_batch.size(0)
            total_addr_loss += addr_loss.item() * x_batch.size(0)
        avg_cmd_loss = total_cmd_loss / len(dataloader.dataset)
        avg_addr_loss = total_addr_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} \\t Cmd loss: {avg_cmd_loss:.4f} \\t Addr loss: {avg_addr_loss:.4f}")

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_addr_mse = 0.0
    with torch.no_grad():
        for x_batch, cmd_label, addr_label in dataloader:
            x_batch = x_batch.to(device)
            cmd_label = cmd_label.to(device)
            addr_label = addr_label.to(device)
            cmd_logits, addr_pred = model(x_batch)
            _, predicted_cmd = torch.max(cmd_logits, dim=1)
            total_correct += (predicted_cmd == cmd_label).sum().item()
            total_samples += cmd_label.size(0)
            total_addr_mse += F.mse_loss(addr_pred, addr_label, reduction='sum').item()
    acc = total_correct / total_samples if total_samples else 0.0
    mse = total_addr_mse / total_samples if total_samples else 0.0
    print(f"Evaluation\\t Cmd accuracy: {acc:.4f} \\t Addr MSE: {mse:.6f}")

def prepare_sequences(num_sequences: int, seq_len: int, read_offset_limit: int = 3) -> List[List[Tuple[int, List[float]]]]:
    gen = ContrastiveDataGenerator(seq_len=seq_len, read_offset_limit=read_offset_limit)
    return [gen._generate_base_valid_sequence() for _ in range(num_sequences)]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train GRU model for NAND sequence generation")
    parser.add_argument('--model_type', type=str, default='GRU', help='select model type')
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of sequences to generate for training')
    parser.add_argument('--seq_len', type=int, default=50, help='Length of each generated sequence')
    parser.add_argument('--lookback', type=int, default=10, help='Number of past events to use for prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='GRU hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sequences = prepare_sequences(args.num_sequences, args.seq_len)
    dataset = GenerativeDataset(sequences, lookback=args.lookback)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    input_dim = dataset[0][0].shape[1] if dataset else 7
#    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
#                dim_feedforward: int = 128, dropout: float = 0.1, max_seq_len: int = 50):

    if args.model_type == 'GRU':
        model = GRUModel(input_dim=input_dim)
    elif args.model_type == 'Transformer':
        model = TransformerModel(input_dim=input_dim)
    elif args.model_type == 'Pointer':
        model = PointerNetwork(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    train(model, dataloader, device, num_epochs=args.epochs, lr=args.lr)
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main()
