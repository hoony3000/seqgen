import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import random_split

# --- 기본 설정 ---
CMD_VOCAB = {'erase': 0, 'program': 1, 'read': 2}
ID_TO_CMD = {v: k for k, v in CMD_VOCAB.items()}
MAX_ADDR = {'die': 1, 'plane': 3, 'block': 999, 'page': 2563}
ADDR_KEYS = ['die', 'plane', 'block', 'page']

# --- 유효성 검사기 ---
class ScenarioValidator:
    """ ScenarioValidator는 NAND 시퀀스의 유효성을 검사하고 상태를 관리합니다.
    :param read_offset_limit: 읽기 명령이 허용되는 최대 페이지 오프셋
    """
    def __init__(self, read_offset_limit=3):
        """
        ScenarioValidator는 NAND 시퀀스의 유효성을 검사하고 상태를 관리합니다.
        :param read_offset_limit: 읽기 명령이 허용되는 최대 페이지 오프셋
        """
        self.state = {}
        self.read_offset_limit = read_offset_limit

    def is_valid(self, cmd_id, addr_vec):
        """
        주어진 명령과 주소 벡터가 유효한지 검사합니다.
        :param cmd_id: 명령 ID (0: erase, 1: program, 2: read)
        :param addr_vec: 주소 벡터 (die, plane, block, page)
        :return: 유효성 여부 (True/False)
        """
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
        """
        주어진 명령과 주소 벡터에 따라 상태를 업데이트합니다.
        :param cmd_id: 명령 ID (0: erase, 1: program, 2: read)
        :param addr_vec: 주소 벡터 (die, plane, block, page)
        """
        key = tuple(round(addr_vec[i] * MAX_ADDR[ADDR_KEYS[i]]) for i in range(3))
        page = round(addr_vec[3] * MAX_ADDR['page'])
        if cmd_id == CMD_VOCAB['erase']:
            self.state[key] = {'pages': set(), 'last_page': -1}
        elif cmd_id == CMD_VOCAB['program']:
            self.state.setdefault(key, {'pages': set(), 'last_page': -1})
            self.state[key]['pages'].add(page)
            self.state[key]['last_page'] = page

# --- Contrastive 데이터 제너레이터 (개선됨) ---
class ContrastiveDataGenerator:
    """ ContrastiveDataGenerator는 유효하고 무효한 NAND 시퀀스를 생성합니다.
    :param seq_len: 생성할 시퀀스의 길이
    :param num_blocks: 활성 블록의 수
    :param read_offset_limit: 읽기 명령이 허용되는 최대 페이지 오프셋
    """
    def __init__(self, seq_len=100, num_blocks=3, read_offset_limit=3):
        """
        ContrastiveDataGenerator는 유효하고 무효한 NAND 시퀀스를 생성합니다.
        :param seq_len: 생성할 시퀀스의 길이
        :param num_blocks: 활성 블록의 수
        :param read_offset_limit: 읽기 명령이 허용되는 최대 페이지 오프셋
        """
        self.seq_len = seq_len
        self.num_blocks = min(num_blocks, MAX_ADDR['block'] + 1)
        self.read_offset_limit = read_offset_limit

    def _addr_to_vector(self, addr_dict):
        """ Converts an address dictionary to a normalized vector.
        :param addr_dict: Address dictionary with keys 'die', 'plane', 'block', 'page'
        :return: Normalized address vector
        """
        return [addr_dict[k] / MAX_ADDR[k] for k in ADDR_KEYS]

    def _vector_to_addr(self, addr_vec):
        """ Converts a normalized address vector back to an address dictionary.
        :param addr_vec: Normalized address vector
        :return: Address dictionary with keys 'die', 'plane', 'block', 'page'
        """
        return {k: round(addr_vec[i] * MAX_ADDR[k]) for i, k in enumerate(ADDR_KEYS)}

    def _get_block_key_from_addr_vec(self, addr_vec):
        """ Extracts the block key from a normalized address vector.
        :param addr_vec: Normalized address vector
        :return: Block key tuple (die, plane, block)
        """
        addr_dict = self._vector_to_addr(addr_vec)
        return tuple(addr_dict[k] for k in ADDR_KEYS[:-1])

    def _generate_base_valid_sequence(self):
        """ Generates a base valid sequence of NAND commands.
        :return: A valid sequence of commands and addresses
        """
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
            cmd_id = -1
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
        """ Generates a valid NAND command sequence.
        :return: A valid sequence of commands and addresses, with label 1
        """
        return self._generate_base_valid_sequence(), 1

    def generate_invalid(self, attempts=20):
        """ Generates an invalid NAND command sequence by corrupting a valid sequence.
        :param attempts: Number of attempts to create an invalid sequence
        :return: A corrupted sequence, label 0, and the corruption strategy used
        """
        for _ in range(attempts):
            # Generate a base valid sequence
            base_seq = self._generate_base_valid_sequence()
            corrupted_seq = copy.deepcopy(base_seq)
            
            # Find pre-corruption state
            validator = ScenarioValidator(self.read_offset_limit)
            states_before_op = []
            for cmd_id, addr_vec in corrupted_seq:
                states_before_op.append(copy.deepcopy(validator.state))
                validator.update_state(cmd_id, addr_vec)

            # Pick a random operation to corrupt (not the first erase)
            idx = random.randint(self.num_blocks, len(corrupted_seq) - 1)
            original_cmd_id, original_addr_vec = corrupted_seq[idx]
            addr_dict = self._vector_to_addr(original_addr_vec)
            block_key = self._get_block_key_from_addr_vec(original_addr_vec)
            state_before = states_before_op[idx].get(block_key)

            if not state_before: continue

            # NEW: Choose a corruption strategy
            corruption_strategy = 'none'
            possible_strategies = []
            if ID_TO_CMD[original_cmd_id] == 'program':
                possible_strategies.append('page_hop')
            if ID_TO_CMD[original_cmd_id] == 'read':
                possible_strategies.extend(['stale_read', 'read_unwritten'])
            
            if not possible_strategies: continue
            corruption_strategy = random.choice(possible_strategies)

            # Apply corruption
            if corruption_strategy == 'page_hop':
                addr_dict['page'] += 1 # Skip a page
            elif corruption_strategy == 'stale_read':
                if state_before['last_page'] >= self.read_offset_limit:
                    addr_dict['page'] = state_before['last_page'] - self.read_offset_limit
                else: continue # Cannot create this error type
            elif corruption_strategy == 'read_unwritten':
                all_pages = set(range(MAX_ADDR['page'] + 1))
                unwritten_pages = all_pages - state_before['pages']
                if unwritten_pages:
                    addr_dict['page'] = random.choice(list(unwritten_pages))
                else: continue # Cannot create this error type

            corrupted_addr_vec = self._addr_to_vector(addr_dict)
            corrupted_seq[idx] = (original_cmd_id, corrupted_addr_vec)

            # Verify it's actually invalid now
            final_validator = ScenarioValidator(self.read_offset_limit)
            is_truly_invalid = False
            for cmd_id, addr_vec in corrupted_seq:
                if not final_validator.is_valid(cmd_id, addr_vec):
                    is_truly_invalid = True
                    break
                final_validator.update_state(cmd_id, addr_vec)
            
            if is_truly_invalid:
                return corrupted_seq, 0, corruption_strategy

        # Fallback
        return self._generate_base_valid_sequence(), 1, 'fallback_valid'

# --- 3D 시각화 함수 ---
def visualize_sequence_3d(sequence, title="NAND Access Trajectory"):
    """ Visualizes a sequence of NAND commands in 3D space.
    :param sequence: List of tuples (cmd_id, addr_vec) representing the sequence
    :param title: Title for the plot
    """
    color_map = {CMD_VOCAB['program']: 'blue', CMD_VOCAB['read']: 'green', CMD_VOCAB['erase']: 'red'}
    block_traces = {}
    for t, (cmd_id, addr_vec) in enumerate(sequence):
        addr = {k: round(addr_vec[i] * MAX_ADDR[k]) for i, k in enumerate(ADDR_KEYS)}
        block = addr['block']
        page = addr['page']
        block_traces.setdefault(block, []).append((page, t, cmd_id))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for blk, pts in block_traces.items():
        if not pts: continue
        pts.sort(key=lambda x: x[1])
        pages, times, cmds = zip(*pts)
        xs = [blk] * len(pages)
        ax.plot(xs, pages, times, alpha=0.4)
        ax.scatter(xs, pages, times, c=[color_map[c] for c in cmds], marker='o')
    ax.set_xlabel('Block'); ax.set_ylabel('Page'); ax.set_zlabel('Time')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# --- 데이터셋 클래스 ---
class NANDSequenceDataset(Dataset):
    """ NAND 시퀀스 데이터셋 """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 시퀀스를 텐서로 변환
        # (cmd_id, addr_vec) -> (cmd_id, die, plane, block, page)
        seq_data = []
        for cmd_id, addr_vec in sequence:
            # cmd_id를 원-핫 인코딩하지 않고 그대로 사용
            # 주소 벡터는 이미 정규화되어 있음
            seq_data.append([cmd_id] + addr_vec)

        return torch.tensor(seq_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# --- 모델 아키텍처 (Structured Event Model) ---
class StructuredEventModel(nn.Module):
    """ Field-wise embedding과 GRU를 사용하는 구조화된 이벤트 시퀀스 모델 """
    def __init__(self, cmd_vocab_size, die_vocab_size, plane_vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout=0.2):
        super(StructuredEventModel, self).__init__()
        
        # 각 필드에 대한 임베딩 레이어
        self.cmd_embedding = nn.Embedding(cmd_vocab_size, embedding_dim)
        self.die_embedding = nn.Embedding(die_vocab_size, embedding_dim)
        self.plane_embedding = nn.Embedding(plane_vocab_size, embedding_dim)
        
        # 연속형 필드를 위한 선형 프로젝션
        self.block_proj = nn.Linear(1, embedding_dim)
        self.page_proj = nn.Linear(1, embedding_dim)
        
        # Concat 후의 전체 입력 차원
        concat_dim = embedding_dim * 5
        
        self.gru = nn.GRU(concat_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, 5) -> [cmd, die, plane, block, page]
        
        # 필드 분리 및 타입 변환
        cmd_ids = x[:, :, 0].long()
        # 정규화된 주소를 다시 정수 인덱스로 변환
        die_ids = (x[:, :, 1] * MAX_ADDR['die']).round().long()
        plane_ids = (x[:, :, 2] * MAX_ADDR['plane']).round().long()
        
        block_vals = x[:, :, 3].unsqueeze(-1)  # (batch, seq_len, 1)
        page_vals = x[:, :, 4].unsqueeze(-1)   # (batch, seq_len, 1)
        
        # 임베딩 및 프로젝션 적용
        cmd_embed = self.cmd_embedding(cmd_ids)
        die_embed = self.die_embedding(die_ids)
        plane_embed = self.plane_embedding(plane_ids)
        block_embed = self.block_proj(block_vals)
        page_embed = self.page_proj(page_vals)
        
        # 임베딩 벡터 연결
        combined = torch.cat([cmd_embed, die_embed, plane_embed, block_embed, page_embed], dim=2)
        
        # GRU에 전달
        gru_out, _ = self.gru(combined)
        last_hidden_state = gru_out[:, -1, :]
        
        out = self.fc(last_hidden_state)
        out = self.sigmoid(out)
        return out

# --- 모델 학습 ---
def train_model(model, dataloader, epochs=20, lr=0.001):
    """ 모델 학습 함수 """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for sequences, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 정확도 계산
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# --- 모델 평가 ---
def evaluate_model(model, dataloader):
    """ 테스트 데이터로 모델을 평가하고 성능 지표를 출력합니다. """
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences).squeeze()
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # 성능 지표 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')

    print("\n--- Model Evaluation Results ---")
    print(f"Confusion Matrix:")
    print(f"  [[TN: {tn}, FP: {fp}]", )
    print(f"   [FN: {fn}, TP: {tp}]]")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------------")

# --- Main ---
def main():
    # --- 데이터 생성 ---
    generator = ContrastiveDataGenerator(seq_len=50, num_blocks=3, read_offset_limit=5)
    num_samples = 1000  # 데이터 샘플 증가
    sequences = []
    labels = []
    print(f"Generating {num_samples * 2} samples...")
    for i in range(num_samples):
        valid_seq, valid_label = generator.generate_valid()
        sequences.append(valid_seq)
        labels.append(valid_label)
        invalid_seq, invalid_label, _ = generator.generate_invalid()
        sequences.append(invalid_seq)
        labels.append(invalid_label)

    # --- 데이터셋 분할 (학습용/테스트용) ---
    full_dataset = NANDSequenceDataset(sequences, labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 모델 초기화 및 학습 ---
    embedding_dim = 16
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = StructuredEventModel(
        cmd_vocab_size=len(CMD_VOCAB),
        die_vocab_size=MAX_ADDR['die'] + 1,
        plane_vocab_size=MAX_ADDR['plane'] + 1,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
    print("\n--- Starting Model Training ---")
    train_model(model, train_dataloader, epochs=25)

    # --- 모델 평가 ---
    evaluate_model(model, test_dataloader)

    # --- 모델 저장 ---
    save_path = "model_250802.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Model saved to {save_path} ---")

    # --- 시각화 (주석 처리) ---
    # print("\n--- Visualizing a few samples ---")
    # # 유효 시퀀스 시각화
    # valid_sequence, valid_label = generator.generate_valid()
    # visualize_sequence_3d(valid_sequence, title="Valid Sequence Trajectory")
    # # 무효 시퀀스 시각화
    # invalid_sequence, invalid_label, strategy = generator.generate_invalid()
    # visualize_sequence_3d(invalid_sequence, title=f"Invalid Sequence Trajectory (via {strategy})")


if __name__ == "__main__":
    main()
