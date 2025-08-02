import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# --- 기본 설정 ---
CMD_VOCAB = {'erase': 0, 'program': 1, 'read': 2}
ID_TO_CMD = {v: k for k, v in CMD_VOCAB.items()}
MAX_ADDR = {'channel': 1, 'chip': 1, 'die': 1, 'plane': 1, 'block': 7, 'page': 15}
ADDR_KEYS = ['channel', 'chip', 'die', 'plane', 'block', 'page']

# --- 유효성 검사기 ---
class ScenarioValidator:
    def __init__(self, read_offset_limit=3):
        self.state = {}
        self.read_offset_limit = read_offset_limit

    def is_valid(self, cmd_id, addr_vec):
        key = tuple(round(addr_vec[i] * MAX_ADDR[ADDR_KEYS[i]]) for i in range(5))
        page = round(addr_vec[5] * MAX_ADDR['page'])

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
        key = tuple(round(addr_vec[i] * MAX_ADDR[ADDR_KEYS[i]]) for i in range(5))
        page = round(addr_vec[5] * MAX_ADDR['page'])
        if cmd_id == CMD_VOCAB['erase']:
            self.state[key] = {'pages': set(), 'last_page': -1}
        elif cmd_id == CMD_VOCAB['program']:
            self.state.setdefault(key, {'pages': set(), 'last_page': -1})
            self.state[key]['pages'].add(page)
            self.state[key]['last_page'] = page

# --- Contrastive 데이터 제너레이터 (개선됨) ---
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
        # (이전과 동일한 로직)
        sequence = []
        block_states = {}
        validator = ScenarioValidator(self.read_offset_limit)
        all_blocks = [(c, ci, d, p, b) for c in range(MAX_ADDR['channel'] + 1) for ci in range(MAX_ADDR['chip'] + 1) for d in range(MAX_ADDR['die'] + 1) for p in range(MAX_ADDR['plane'] + 1) for b in range(MAX_ADDR['block'] + 1)]
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
        return self._generate_base_valid_sequence(), 1

    def generate_invalid(self, attempts=20):
        for _ in range(attempts):
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
    # (이전과 동일)
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

# --- Main ---
def main():
    generator = ContrastiveDataGenerator(seq_len=100, num_blocks=4, read_offset_limit=5)

    # 1. 유효 시퀀스 생성 및 시각화
    valid_sequence, valid_label = generator.generate_valid()
    print(f"--- Generated Valid Sequence (Label: {valid_label}) ---")
    visualize_sequence_3d(valid_sequence, title="Valid Sequence Trajectory")

    # 2. 무효 시퀀스 생성 및 시각화
    invalid_sequence, invalid_label, strategy = generator.generate_invalid()
    print(f"\n--- Generated Invalid Sequence (Label: {invalid_label}, Strategy: '{strategy}') ---")
    visualize_sequence_3d(invalid_sequence, title=f"Invalid Sequence Trajectory (via {strategy})")
    
    # 실패 지점 검증
    validator = ScenarioValidator()
    for i, (cmd_id, addr_vec) in enumerate(invalid_sequence):
        if not validator.is_valid(cmd_id, addr_vec):
            addr = {k: round(addr_vec[i] * MAX_ADDR[k]) for i, k in enumerate(ADDR_KEYS)}
            print(f"  Invalid command at index {i}: {ID_TO_CMD[cmd_id]:<7} | Addr: {addr}")
            break
        validator.update_state(cmd_id, addr_vec)
    else:
        if invalid_label == 0: print("  Warning: Corrupted sequence was unexpectedly valid.")

if __name__ == "__main__":
    main()