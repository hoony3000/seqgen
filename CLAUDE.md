# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NAND flash memory sequence generator and simulation framework that combines:
- NAND flash memory operation simulation with timing models
- Machine learning models for sequence validation and generation
- PyTorch-based classifier and generative models for NAND operation sequences

## Key Commands

### Installation
```bash
pip install -e .
```

### Running the Main NAND Sequence Generator
```bash
python main.py
```

### Running ML Training and Evaluation
```bash
# Train both classifier and generator models (default)
python seq_sim.py

# Train only classifier
python seq_sim.py --model_type classifier

# Train only generator  
python seq_sim.py --model_type generator

# Custom parameters
python seq_sim.py --seq_len 100 --num_samples 5000 --epochs 200 --lr 0.0005
```

### Key Parameters for seq_sim.py
- `--model_type`: 'classifier', 'generator', or 'both' (default: 'both')
- `--seq_len`: Length of command sequences (default: 50)
- `--num_samples`: Number of training samples (default: 2000)
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch sizes for models (default: 1024)

## Architecture Overview

### Core Components

1. **NAND Simulation Framework** (`gen_seq.py`)
   - `DeduplicateBase`: Base class preventing duplicate instance creation
   - `StateSeq`: Represents timing sequences and states 
   - `Operation`: Defines NAND operations (erase, program, read) with timing
   - `NANDScheduler`: Manages multi-die, multi-plane operation scheduling

2. **ML Models** (`seq_sim.py`)
   - `ScenarioValidator`: Validates NAND operation sequences based on flash memory constraints
   - `SequenceDataGenerator`: Generates training data with valid/invalid sequences
   - `NANDSequenceDataset`/`GenerativeNANDSequenceDataset`: PyTorch datasets
   - `Encoder`/`Decoder`: Sequence-to-sequence models with attention
   - `ClassifierModel`: Binary classifier for sequence validity

### NAND Operation Types
- Erase operations (block-level)
- Program operations (page-level: LSB, CSB, MSB)
- Read operations (page-level: LSB, CSB, MSB)
- Status register reads
- Reset operations
- Suspend/resume operations

### Configuration
- `config.yaml`: Contains timing specifications (`TimeSpec`) and execution times (`ExecTime`)
- Timing values in nanoseconds (constants: NS, US, MS, S)

## Model Files
- `*_latest.pth`: Latest trained model weights
- `*_250802.pth`: Dated model checkpoints
- Models: classifier, generative encoder/decoder

## Key Constraints and Validation Rules
1. Erase must happen before program operations on a block
2. Program operations must be sequential (page order)
3. Read operations can only access programmed pages
4. Read offset limits apply for temporal locality

## Development Notes
- Uses PyTorch for ML components
- YAML configuration for timing parameters
- Thread-safe instance management with deduplication
- Supports multi-die, multi-plane NAND architectures
- Feature extraction includes block states, timing, and access patterns