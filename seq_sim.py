import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# --- 기본 설정 ---
CMD_VOCAB = {'erase': 0, 'program': 1, 'read': 2}
ID_TO_CMD = {v: k for v, k in CMD_VOCAB.items()}
MAX_ADDR = {'die': 1, 'plane': 3, 'block': 999, 'page': 2563}
ADDR_KEYS = ['die', 'plane', 'block', 'page']

# --- 유효성 검사기 ---
class ScenarioValidator:
    def __init__(self, read_offset_limit=3):
        self.state = {}
        self.read_offset_limit = read_offset_limit

    def is_valid(self, cmd_id, addr_vec):
        key = tuple(round(addr_vec[i] * MAX_ADDR[ADDR_KEYS[i]]) for i in range(3))
        page = round(addr_vec[3] * MAX_ADDR['page'])
        if cmd_id == CMD_VOCAB['erase']:
            return True
        if cmd_id == CMD_VOCAB['program']:
            if key not in self.state: return False
            last_page = self.state[key]['last_page']
            return page not in self.state[key]['pages'] and page == last_page + 1
        if cmd_id == CMD_VOCAB['read']:
            if key not in self.state: return False
            last_page = self.state[key]['last_page']
            return page in self.state[key]['pages'] and (last_page - page < self.read_offset_limit)
        return False

    def update_state(self, cmd_id, addr_vec):
        key = tuple(round(addr_vec[i] * MAX_ADDR[ADDR_KEYS[i]]) for i in range(3))
        page = round(addr_vec[3] * MAX_ADDR['page'])
        if cmd_id == CMD_VOCAB['erase']:
            self.state[key] = {'pages': set(), 'last_page': -1}
        elif cmd_id == CMD_VOCAB['program']:
            self.state.setdefault(key, {'pages': set(), 'last_page': -1})
            self.state[key]['pages'].add(page)
            self.state[key]['last_page'] = page

# --- 데이터 제너레이터 ---
class ContrastiveDataGenerator:
    def __init__(self, seq_len=100, num_blocks=3, read_offset_limit=3):
        self.seq_len = seq_len
        self.num_blocks = min(num_blocks, MAX_ADDR['block'] + 1)
        self.read_offset_limit = read_offset_limit

    def _addr_to_vector(self, addr_dict):
        return [addr_dict[k] / MAX_ADDR[k] for k in ADDR_KEYS]

    def _vector_to_addr(self, addr_vec):
        return {k: round(addr_vec[i] * MAX_ADDR[k]) for i, k in enumerate(ADDR_KEYS)}

    def _get_block_key_from_addr_vec(self, addr_vec):
        addr_dict = self._vector_to_addr(addr_vec)
        return tuple(addr_dict[k] for k in ADDR_KEYS[:-1])

    def _generate_base_valid_sequence(self):
        sequence = []
        block_states = {}
        validator = ScenarioValidator(self.read_offset_limit)
        all_blocks = [(d, p, b) for d in range(MAX_ADDR['die'] + 1) for p in range(MAX_ADDR['plane'] + 1) for b in range(MAX_ADDR['block'] + 1)]
        active_blocks = random.sample(all_blocks, self.num_blocks)
        for block_key in active_blocks:
            addr = dict(zip(ADDR_KEYS[:-1], block_key)); addr['page'] = 0
            addr_vec = self._addr_to_vector(addr)
            cmd_id = CMD_VOCAB['erase']
            validator.update_state(cmd_id, addr_vec)
            block_states[block_key] = {'last_page': -1, 'programmed_pages': set()}
            sequence.append((cmd_id, addr_vec))
        while len(sequence) < self.seq_len:
            block_key = random.choice(active_blocks)
            state = block_states[block_key]
            possible_actions = []
            if state['last_page'] < MAX_ADDR['page']: possible_actions.append('program')
            if state['last_page'] >= 0: possible_actions.append('read')
            if not possible_actions: possible_actions.append('erase')
            action = random.choice(possible_actions)
            addr = dict(zip(ADDR_KEYS[:-1], block_key))
            if action == 'erase':
                addr['page'] = 0
                cmd_id = CMD_VOCAB['erase']
                block_states[block_key] = {'last_page': -1, 'programmed_pages': set()}
            elif action == 'program':
                next_page = state['last_page'] + 1
                addr['page'] = next_page
                cmd_id = CMD_VOCAB['program']
                state['last_page'] = next_page
                state['programmed_pages'].add(next_page)
            elif action == 'read':
                last_page = state['last_page']
                start_page = max(0, last_page - self.read_offset_limit + 1)
                addr['page'] = random.randint(start_page, last_page)
                cmd_id = CMD_VOCAB['read']
            addr_vec = self._addr_to_vector(addr)
            validator.update_state(cmd_id, addr_vec)
            sequence.append((cmd_id, addr_vec))
        return sequence[:self.seq_len]

    def generate_valid(self):
        return self._generate_base_valid_sequence(), 1

    def generate_invalid(self, attempts=20):
        for _ in range(attempts):
            base_seq = self._generate_base_valid_sequence()
            corrupted_seq = copy.deepcopy(base_seq)
            validator = ScenarioValidator(self.read_offset_limit)
            states_before_op = [copy.deepcopy(validator.state)]
            for cmd_id, addr_vec in corrupted_seq:
                validator.update_state(cmd_id, addr_vec)
                states_before_op.append(copy.deepcopy(validator.state))
            idx = random.randint(self.num_blocks, len(corrupted_seq) - 1)
            original_cmd_id, original_addr_vec = corrupted_seq[idx]
            addr_dict = self._vector_to_addr(original_addr_vec)
            block_key = self._get_block_key_from_addr_vec(original_addr_vec)
            state_before = states_before_op[idx].get(block_key)
            if not state_before: continue
            corruption_strategy = random.choice(['page_hop', 'stale_read', 'read_unwritten'])
            if corruption_strategy == 'page_hop':
                addr_dict['page'] += 1
            elif corruption_strategy == 'stale_read':
                if state_before['last_page'] >= self.read_offset_limit:
                    addr_dict['page'] = state_before['last_page'] - self.read_offset_limit
                else: continue
            elif corruption_strategy == 'read_unwritten':
                unwritten_pages = set(range(MAX_ADDR['page'] + 1)) - state_before['pages']
                if unwritten_pages:
                    addr_dict['page'] = random.choice(list(unwritten_pages))
                else: continue
            corrupted_addr_vec = self._addr_to_vector(addr_dict)
            corrupted_seq[idx] = (original_cmd_id, corrupted_addr_vec)
            final_validator = ScenarioValidator(self.read_offset_limit)
            is_truly_invalid = any(not final_validator.is_valid(cmd, vec) for cmd, vec in corrupted_seq)
            if is_truly_invalid:
                return corrupted_seq, 0, corruption_strategy
        return self._generate_base_valid_sequence(), 1, 'fallback_valid'

# --- 상태 기반 특징 추출기 ---
class StatefulFeatureExtractor:
    def __init__(self, max_addr, cmd_vocab, read_offset_limit):
        self.max_addr = max_addr
        self.cmd_vocab = cmd_vocab
        self.read_offset_limit = read_offset_limit
        self.block_states = {}

    def _get_block_key_from_addr_vec(self, addr_vec):
        addr_dict = {k: round(addr_vec[i] * self.max_addr[k]) for i, k in enumerate(ADDR_KEYS)}
        return tuple(addr_dict[k] for k in ADDR_KEYS[:-1])

    def extract_features_and_update_state(self, cmd_id, addr_vec):
        block_key = self._get_block_key_from_addr_vec(addr_vec)
        current_page = round(addr_vec[3] * self.max_addr['page'])

        if block_key not in self.block_states:
            self.block_states[block_key] = {'last_page': -1, 'programmed_pages': set()}

        current_block_state = self.block_states[block_key]

        is_block_erased_feature = 1.0 if current_block_state['last_page'] == -1 else 0.0
        is_page_written_feature = 1.0 if current_page in current_block_state['programmed_pages'] else 0.0

        is_next_page_expected_feature = 0.0
        is_read_offset_valid_feature = 0.0

        if cmd_id == self.cmd_vocab['program']:
            if current_page == current_block_state['last_page'] + 1:
                is_next_page_expected_feature = 1.0
        elif cmd_id == self.cmd_vocab['read']:
            if is_page_written_feature == 1.0 and \
               (current_block_state['last_page'] - current_page < self.read_offset_limit):
                is_read_offset_valid_feature = 1.0

        # Update state based on current command
        if cmd_id == self.cmd_vocab['erase']:
            self.block_states[block_key] = {'last_page': -1, 'programmed_pages': set()}
        elif cmd_id == self.cmd_vocab['program']:
            self.block_states[block_key]['programmed_pages'].add(current_page)
            self.block_states[block_key]['last_page'] = current_page

        return [is_block_erased_feature, is_page_written_feature,
                is_next_page_expected_feature, is_read_offset_valid_feature]

# --- 데이터셋 클래스 (분류 모델용) ---
class NANDSequenceDataset(Dataset):
    def __init__(self, sequences, labels, strategies=None):
        self.sequences = sequences
        self.labels = labels
        self.strategies = strategies if strategies is not None else ['valid'] * len(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        original_sequence, label, strategy = self.sequences[idx], self.labels[idx], self.strategies[idx]
        
        feature_extractor = StatefulFeatureExtractor(MAX_ADDR, CMD_VOCAB, 5) # read_offset_limit from main
        
        enriched_seq_data = []
        for cmd_id, addr_vec in original_sequence:
            new_features = feature_extractor.extract_features_and_update_state(cmd_id, addr_vec)
            enriched_seq_data.append([cmd_id] + addr_vec + new_features)

        return torch.tensor(enriched_seq_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), strategy

# --- 데이터셋 클래스 (생성 모델용으로 수정) ---
class GenerativeNANDSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = []
        for seq in sequences:
            # 각 시퀀스에서 (t-1)까지의 입력과 t번째 타겟을 생성
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                target_cmd = seq[i][0] # t번째 명령
                target_addr = torch.tensor(seq[i][1], dtype=torch.float32) # t번째 주소 벡터
                self.sequences.append((input_seq, target_cmd, target_addr))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_cmd, target_addr = self.sequences[idx]
        
        # StatefulFeatureExtractor를 사용하여 입력 시퀀스 풍부화
        feature_extractor = StatefulFeatureExtractor(MAX_ADDR, CMD_VOCAB, 5) # read_offset_limit from main
        enriched_input_seq = []
        for cmd_id, addr_vec in input_seq:
            new_features = feature_extractor.extract_features_and_update_state(cmd_id, addr_vec)
            enriched_input_seq.append([cmd_id] + addr_vec + new_features)

        return torch.tensor(enriched_input_seq, dtype=torch.float32), target_cmd, target_addr

# --- 모델 아키텍처 (인코더-디코더) ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_outputs):
        attn_weights = self.attn(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), gru_outputs).squeeze(1)
        return context_vector

class Encoder(nn.Module):
    def __init__(self, cmd_vocab_size, die_vocab_size, plane_vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.4):
        super(Encoder, self).__init__()
        self.cmd_embedding = nn.Embedding(cmd_vocab_size, embedding_dim)
        self.die_embedding = nn.Embedding(die_vocab_size, embedding_dim)
        self.plane_embedding = nn.Embedding(plane_vocab_size, embedding_dim)
        self.block_proj = nn.Linear(1, embedding_dim)
        self.page_proj = nn.Linear(1, embedding_dim)
        self.is_block_erased_proj = nn.Linear(1, embedding_dim)
        self.is_page_written_proj = nn.Linear(1, embedding_dim)
        self.is_next_page_expected_proj = nn.Linear(1, embedding_dim)
        self.is_read_offset_valid_proj = nn.Linear(1, embedding_dim)
        
        concat_dim = embedding_dim * 9
        self.gru = nn.GRU(concat_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)

    def forward(self, x):
        cmd_ids = x[:, :, 0].long()
        die_ids = (x[:, :, 1] * MAX_ADDR['die']).round().long()
        plane_ids = (x[:, :, 2] * MAX_ADDR['plane']).round().long()
        block_vals = x[:, :, 3].unsqueeze(-1)
        page_vals = x[:, :, 4].unsqueeze(-1)
        is_block_erased_vals = x[:, :, 5].unsqueeze(-1)
        is_page_written_vals = x[:, :, 6].unsqueeze(-1)
        is_next_page_expected_vals = x[:, :, 7].unsqueeze(-1)
        is_read_offset_valid_vals = x[:, :, 8].unsqueeze(-1)
        
        cmd_embed = self.cmd_embedding(cmd_ids)
        die_embed = self.die_embedding(die_ids)
        plane_embed = self.plane_embedding(plane_ids)
        block_embed = self.block_proj(block_vals)
        page_embed = self.page_proj(page_vals)
        is_block_erased_embed = self.is_block_erased_proj(is_block_erased_vals)
        is_page_written_embed = self.is_page_written_proj(is_page_written_vals)
        is_next_page_expected_embed = self.is_next_page_expected_proj(is_next_page_expected_vals)
        is_read_offset_valid_embed = self.is_read_offset_valid_proj(is_read_offset_valid_vals)
        
        combined = torch.cat([cmd_embed, die_embed, plane_embed, block_embed, page_embed,
                              is_block_erased_embed, is_page_written_embed,
                              is_next_page_expected_embed, is_read_offset_valid_embed], dim=2)
        
        gru_out, hidden = self.gru(combined)
        context_vector = self.attention(gru_out)
        return context_vector, hidden

class Decoder(nn.Module):
    def __init__(self, cmd_vocab_size, die_vocab_size, plane_vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.4):
        super(Decoder, self).__init__()
        self.cmd_embedding = nn.Embedding(cmd_vocab_size, embedding_dim)
        self.die_embedding = nn.Embedding(die_vocab_size, embedding_dim)
        self.plane_embedding = nn.Embedding(plane_vocab_size, embedding_dim)
        self.block_proj = nn.Linear(1, embedding_dim)
        self.page_proj = nn.Linear(1, embedding_dim)
        self.is_block_erased_proj = nn.Linear(1, embedding_dim)
        self.is_page_written_proj = nn.Linear(1, embedding_dim)
        self.is_next_page_expected_proj = nn.Linear(1, embedding_dim)
        self.is_read_offset_valid_proj = nn.Linear(1, embedding_dim)

        concat_dim = embedding_dim * 9 # Same as encoder input
        self.gru = nn.GRU(concat_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.cmd_output = nn.Linear(hidden_size, cmd_vocab_size)
        self.addr_output = nn.Linear(hidden_size, len(ADDR_KEYS)) # Output normalized address vector

    def forward(self, x, hidden):
        # x is the raw feature vector (batch, 1, 9)
        cmd_ids = x[:, :, 0].long()
        die_ids = (x[:, :, 1] * MAX_ADDR['die']).round().long()
        plane_ids = (x[:, :, 2] * MAX_ADDR['plane']).round().long()
        block_vals = x[:, :, 3].unsqueeze(-1)
        page_vals = x[:, :, 4].unsqueeze(-1)
        is_block_erased_vals = x[:, :, 5].unsqueeze(-1)
        is_page_written_vals = x[:, :, 6].unsqueeze(-1)
        is_next_page_expected_vals = x[:, :, 7].unsqueeze(-1)
        is_read_offset_valid_vals = x[:, :, 8].unsqueeze(-1)
        
        cmd_embed = self.cmd_embedding(cmd_ids)
        die_embed = self.die_embedding(die_ids)
        plane_embed = self.plane_embedding(plane_ids)
        block_embed = self.block_proj(block_vals)
        page_embed = self.page_proj(page_vals)
        is_block_erased_embed = self.is_block_erased_proj(is_block_erased_vals)
        is_page_written_embed = self.is_page_written_proj(is_page_written_vals)
        is_next_page_expected_embed = self.is_next_page_expected_proj(is_next_page_expected_vals)
        is_read_offset_valid_embed = self.is_read_offset_valid_proj(is_read_offset_valid_vals)
        
        combined = torch.cat([cmd_embed, die_embed, plane_embed, block_embed, page_embed,
                              is_block_erased_embed, is_page_written_embed,
                              is_next_page_expected_embed, is_read_offset_valid_embed], dim=2)
        
        # combined is now (batch, 1, 144)
        gru_out, hidden = self.gru(combined, hidden)
        
        cmd_pred = self.cmd_output(gru_out.squeeze(1)) # (batch, cmd_vocab_size)
        addr_pred = self.addr_output(gru_out.squeeze(1)) # (batch, len(ADDR_KEYS))
        
        return cmd_pred, addr_pred, hidden

class StructuredEventModel(nn.Module):
    def __init__(self, cmd_vocab_size, die_vocab_size, plane_vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout=0.4):
        super(StructuredEventModel, self).__init__()
        
        self.cmd_embedding = nn.Embedding(cmd_vocab_size, embedding_dim)
        self.die_embedding = nn.Embedding(die_vocab_size, embedding_dim)
        self.plane_embedding = nn.Embedding(plane_vocab_size, embedding_dim)
        self.block_proj = nn.Linear(1, embedding_dim)
        self.page_proj = nn.Linear(1, embedding_dim)
        
        # New feature projections
        self.is_block_erased_proj = nn.Linear(1, embedding_dim)
        self.is_page_written_proj = nn.Linear(1, embedding_dim)
        self.is_next_page_expected_proj = nn.Linear(1, embedding_dim)
        self.is_read_offset_valid_proj = nn.Linear(1, embedding_dim)
        
        concat_dim = embedding_dim * 9 # 5 original + 4 new features
        
        self.gru = nn.GRU(concat_dim, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cmd_ids = x[:, :, 0].long()
        die_ids = (x[:, :, 1] * MAX_ADDR['die']).round().long()
        plane_ids = (x[:, :, 2] * MAX_ADDR['plane']).round().long()
        block_vals = x[:, :, 3].unsqueeze(-1)
        page_vals = x[:, :, 4].unsqueeze(-1)
        
        is_block_erased_vals = x[:, :, 5].unsqueeze(-1)
        is_page_written_vals = x[:, :, 6].unsqueeze(-1)
        is_next_page_expected_vals = x[:, :, 7].unsqueeze(-1)
        is_read_offset_valid_vals = x[:, :, 8].unsqueeze(-1)
        
        cmd_embed = self.cmd_embedding(cmd_ids)
        die_embed = self.die_embedding(die_ids)
        plane_embed = self.plane_embedding(plane_ids)
        block_embed = self.block_proj(block_vals)
        page_embed = self.page_proj(page_vals)
        
        is_block_erased_embed = self.is_block_erased_proj(is_block_erased_vals)
        is_page_written_embed = self.is_page_written_proj(is_page_written_vals)
        is_next_page_expected_embed = self.is_next_page_expected_proj(is_next_page_expected_vals)
        is_read_offset_valid_embed = self.is_read_offset_valid_proj(is_read_offset_valid_vals)
        
        combined = torch.cat([cmd_embed, die_embed, plane_embed, block_embed, page_embed,
                              is_block_erased_embed, is_page_written_embed,
                              is_next_page_expected_embed, is_read_offset_valid_embed], dim=2)
        
        gru_out, _ = self.gru(combined)
        context_vector = self.attention(gru_out)
        out = self.fc(context_vector)
        return self.sigmoid(out)

# --- 모델 학습 (가중치 감쇠 및 조기 종료 추가) ---
def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.0001, weight_decay=1e-4, patience=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for sequences, labels, _ in train_loader: # Added _ for strategy
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for sequences, labels, _ in val_loader: # Added _ for strategy
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)

# --- 생성 모델 학습 ---
def train_generative_model(encoder, decoder, train_loader, val_loader, device, epochs=100, lr=0.0001, weight_decay=1e-4, patience=10):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    cmd_criterion = nn.CrossEntropyLoss()
    addr_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_encoder_state = None
    best_decoder_state = None

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for input_sequences, target_cmds, target_addrs in train_loader:
            input_sequences, target_cmds, target_addrs = input_sequences.to(device), target_cmds.to(device), target_addrs.to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Encoder forward
            context_vector, encoder_hidden = encoder(input_sequences)
            
            # Decoder initial input (e.g., start token or first actual token)
            # For simplicity, we'll use the last element of the input sequence as the initial decoder input
            # This might need refinement for true sequence generation from scratch
            decoder_input = input_sequences[:, -1, :].unsqueeze(1) # (batch, 1, input_dim)
            decoder_hidden = encoder_hidden # Use encoder's last hidden state as decoder's initial hidden state

            cmd_pred, addr_pred, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            cmd_loss = cmd_criterion(cmd_pred, target_cmds)
            addr_loss = addr_criterion(addr_pred, target_addrs)
            
            loss = cmd_loss + addr_loss # Combine losses
            total_loss += loss.item()

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)

        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for input_sequences, target_cmds, target_addrs in val_loader:
                input_sequences, target_cmds, target_addrs = input_sequences.to(device), target_cmds.to(device), target_addrs.to(device)
                context_vector, encoder_hidden = encoder(input_sequences)
                decoder_input = input_sequences[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden
                cmd_pred, addr_pred, _ = decoder(decoder_input, decoder_hidden)
                
                cmd_loss = cmd_criterion(cmd_pred, target_cmds)
                addr_loss = addr_criterion(addr_pred, target_addrs)
                val_loss += (cmd_loss + addr_loss).item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            encoder.load_state_dict(best_encoder_state)
            decoder.load_state_dict(best_decoder_state)
            break
    
    if best_encoder_state and best_decoder_state:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)

# --- 모델 평가 (분류 모델용) ---
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels, all_predictions, all_strategies = [], [], []
    with torch.no_grad():
        for sequences, labels, strategies in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            # Handle case where batch size is 1 and squeeze removes all dimensions
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_strategies.extend(strategies)
    
    if not all_labels:
        print("No data to evaluate.")
        return

    # 전체 성능 지표
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)

    print("\n--- Overall Model Evaluation Results (Classifier) ---")
    print(f"Confusion Matrix: [[TN: {tn}, FP: {fp}], [FN: {fn}, TP: {tp}]]")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------------")

    # 오류 유형별 성능 지표
    print("\n--- Evaluation by Corruption Strategy (Classifier) ---")
    unique_strategies = sorted(list(set(all_strategies)))
    for strategy in unique_strategies:
        strategy_indices = [i for i, s in enumerate(all_strategies) if s == strategy]
        if not strategy_indices: continue

        strategy_labels = [all_labels[i] for i in strategy_indices]
        strategy_predictions = [all_predictions[i] for i in strategy_indices]

        acc = accuracy_score(strategy_labels, strategy_predictions)
        
        pos_label = 1 if strategy == 'valid' else 0
        
        prec, rec, f1_s, _ = precision_recall_fscore_support(strategy_labels, strategy_predictions, average='binary', pos_label=pos_label, zero_division=0)

        print(f"Strategy: {strategy:<15} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1_s:.4f} (Metrics for class {pos_label})")
    print("--------------------------------")

# --- 생성 모델용 Collate 함수 ---
def pad_collate_fn(batch):
    # Sort batch by sequence length for potential use with PackedSequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, target_cmds, target_addrs = zip(*batch)
    
    # Pad sequences
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Stack targets
    target_cmds = torch.tensor(target_cmds, dtype=torch.long)
    target_addrs = torch.stack(target_addrs)
    
    return padded_sequences, target_cmds, target_addrs

# --- 시퀀스 생성 함수 ---
def generate_sequence(encoder, decoder, start_sequence, seq_len, validator, feature_extractor, device):
    encoder.eval()
    decoder.eval()
    generated_sequence = copy.deepcopy(start_sequence)
    current_validator_state = copy.deepcopy(feature_extractor.block_states) # Capture initial state

    with torch.no_grad():
        for _ in range(seq_len - len(start_sequence)):
            # Prepare input for encoder
            input_tensor = []
            temp_feature_extractor = StatefulFeatureExtractor(MAX_ADDR, CMD_VOCAB, 5) # Re-initialize for each generation step
            temp_feature_extractor.block_states = copy.deepcopy(current_validator_state) # Restore state

            for cmd_id, addr_vec in generated_sequence:
                new_features = temp_feature_extractor.extract_features_and_update_state(cmd_id, addr_vec)
                input_tensor.append([cmd_id] + addr_vec + new_features)
            
            input_tensor = torch.tensor([input_tensor], dtype=torch.float32).to(device)

            context_vector, encoder_hidden = encoder(input_tensor)
            
            # Decoder input: use the last generated event's features
            decoder_input = input_tensor[:, -1, :].unsqueeze(1)
            decoder_hidden = encoder_hidden

            cmd_pred, addr_pred, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Sample command and address
            predicted_cmd_id = torch.argmax(cmd_pred, dim=1).item()
            predicted_addr_vec = addr_pred.squeeze(0).cpu().numpy()

            # Validate and append
            # For generation, we might want to sample valid commands/addresses
            # For now, just append and let the validator check later
            generated_sequence.append((predicted_cmd_id, predicted_addr_vec.tolist()))
            
            # Update validator state for next step
            current_validator_state = copy.deepcopy(temp_feature_extractor.block_states)

    return generated_sequence

# --- Main ---
def main():
    # --- Device 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # --- 데이터 생성 (분류 모델용) ---
    generator = ContrastiveDataGenerator(seq_len=50, num_blocks=3, read_offset_limit=5)
    num_samples = 2000
    sequences_classifier, labels_classifier, strategies_classifier = [], [], []
    print(f"Generating {num_samples} samples for classifier...")
    for _ in range(num_samples // 2):
        valid_seq, valid_label = generator.generate_valid()
        sequences_classifier.append(valid_seq); labels_classifier.append(valid_label); strategies_classifier.append('valid')
        invalid_seq, invalid_label, corruption_strategy = generator.generate_invalid()
        sequences_classifier.append(invalid_seq); labels_classifier.append(invalid_label); strategies_classifier.append(corruption_strategy)

    full_dataset_classifier = NANDSequenceDataset(sequences_classifier, labels_classifier, strategies_classifier)
    train_size_cls = int(0.7 * len(full_dataset_classifier))
    val_size_cls = int(0.15 * len(full_dataset_classifier))
    test_size_cls = len(full_dataset_classifier) - train_size_cls - val_size_cls
    train_dataset_cls, val_dataset_cls, test_dataset_cls = random_split(full_dataset_classifier, [train_size_cls, val_size_cls, test_size_cls])

    train_loader_cls = DataLoader(train_dataset_cls, batch_size=32, shuffle=True)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=32, shuffle=False)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=32, shuffle=False)

    classifier_model = StructuredEventModel(
        cmd_vocab_size=len(CMD_VOCAB),
        die_vocab_size=MAX_ADDR['die'] + 1,
        plane_vocab_size=MAX_ADDR['plane'] + 1,
        embedding_dim=16, hidden_size=64, num_layers=2, output_size=1, dropout=0.4
    ).to(device)
    
    print("\n--- Starting Classifier Model Training ---")
    train_model(classifier_model, train_loader_cls, val_loader_cls, device, epochs=100, lr=0.0001, weight_decay=1e-4, patience=10)

    print("\n--- Evaluating Classifier Model on Test Set ---")
    evaluate_model(classifier_model, test_loader_cls, device)

    save_path_classifier = "classifier_model_250802.pth"
    torch.save(classifier_model.state_dict(), save_path_classifier)
    print(f"\n--- Best classifier model saved to {save_path_classifier} ---")

    # --- 데이터 생성 (생성 모델용) ---
    sequences_generative = []
    print(f"Generating {num_samples} samples for generative model...")
    for _ in range(num_samples):
        valid_seq, _ = generator.generate_valid()
        sequences_generative.append(valid_seq)
    
    generative_dataset = GenerativeNANDSequenceDataset(sequences_generative)
    gen_train_size = int(0.8 * len(generative_dataset))
    gen_val_size = len(generative_dataset) - gen_train_size
    gen_train_dataset, gen_val_dataset = random_split(generative_dataset, [gen_train_size, gen_val_size])

    gen_train_loader = DataLoader(gen_train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    gen_val_loader = DataLoader(gen_val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    # --- 생성 모델 초기화 및 학습 ---
    embedding_dim_gen = 16
    hidden_size_gen = 64
    num_layers_gen = 2

    encoder = Encoder(
        cmd_vocab_size=len(CMD_VOCAB),
        die_vocab_size=MAX_ADDR['die'] + 1,
        plane_vocab_size=MAX_ADDR['plane'] + 1,
        embedding_dim=embedding_dim_gen,
        hidden_size=hidden_size_gen,
        num_layers=num_layers_gen,
        dropout=0.4
    ).to(device)

    decoder = Decoder(
        cmd_vocab_size=len(CMD_VOCAB),
        die_vocab_size=MAX_ADDR['die'] + 1,
        plane_vocab_size=MAX_ADDR['plane'] + 1,
        embedding_dim=embedding_dim_gen,
        hidden_size=hidden_size_gen,
        num_layers=num_layers_gen,
        dropout=0.4
    ).to(device)

    print("\n--- Starting Generative Model Training ---")
    train_generative_model(encoder, decoder, gen_train_loader, gen_val_loader, device, epochs=50, lr=0.0001, weight_decay=1e-4, patience=10)

    save_path_generative_encoder = "generative_encoder_250802.pth"
    save_path_generative_decoder = "generative_decoder_250802.pth"
    torch.save(encoder.state_dict(), save_path_generative_encoder)
    torch.save(decoder.state_dict(), save_path_generative_decoder)
    print(f"\n--- Best generative models saved to {save_path_generative_encoder} and {save_path_generative_decoder} ---")

    # --- 생성된 시퀀스 검증 (옵션) ---
    print("\n--- Generating and Validating Sample Sequences ---")
    num_generated_sequences = 5
    generated_seq_len = 10 # Shorter for demonstration
    validator = ScenarioValidator(read_offset_limit=5)

    for i in range(num_generated_sequences):
        # Start with a random valid initial command
        initial_cmd_id = random.choice(list(CMD_VOCAB.values()))
        initial_addr_vec = [random.random() for _ in range(len(ADDR_KEYS))]
        start_sequence = [(initial_cmd_id, initial_addr_vec)]

        # Generate sequence
        generated_seq = generate_sequence(encoder, decoder, start_sequence, generated_seq_len, validator, StatefulFeatureExtractor(MAX_ADDR, CMD_VOCAB, 5), device)
        
        # Validate generated sequence
        is_valid = True
        temp_validator = ScenarioValidator(read_offset_limit=5)
        for cmd_id, addr_vec in generated_seq:
            if not temp_validator.is_valid(cmd_id, addr_vec):
                is_valid = False
                break
            temp_validator.update_state(cmd_id, addr_vec)
        
        print(f"Generated Sequence {i+1} (Length: {len(generated_seq)}): {'Valid' if is_valid else 'Invalid'}")
        # print(generated_seq) # Uncomment to see the raw sequence

if __name__ == "__main__":
    main()