# LLM Data Center RL

A comprehensive reinforcement learning framework for optimizing KV-cache compression strategies in Large Language Model data centers.

## Features

- **Unified Environment**: Support for both synthetic and realistic simulation modes
- **Multiple RL Algorithms**: PPO, A2C, DQN, SAC, TD3 with unified interface
- **Baseline Agents**: Rule-based agents for comparison
- **Flexible Configuration**: Easy-to-use configuration system with predefined setups
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Optimized Performance**: Binary search for O(log n) data lookups in realistic mode

## Quick Start

### 1. Training

```bash
python scripts/train.py --config realistic_mode --algorithms PPO DQN

python scripts/train.py --config synthetic_mode --algorithms PPO DQN
```

### 2. Evaluation

```bash
# Evaluate trained models
python scripts/evaluate.py \
    --models-dir results/my_experiment/models \
    --config realistic_mode \
    --n-eval-episodes 10
```

## Data Format

### Invocations File (CSV)

```csv
Time,TIMESTAMP
2024-05-10 00:00:00,150
2024-05-10 00:00:10,145
2024-05-10 00:00:20,160
...
```

- `Time`: Timestamp in YYYY-MM-DD HH:MM:SS format
- `TIMESTAMP`: Number of requests in the time interval

### Energy Price File (CSV)

```csv
INTERVALSTARTTIME_GMT,VALUE
2024-05-10 00:00:00,45.23
2024-05-10 00:15:00,47.15
2024-05-10 00:30:00,44.80
...
```

- `INTERVALSTARTTIME_GMT`: Timestamp in YYYY-MM-DD HH:MM:SS format
- `VALUE`: Energy price in $/MWh

### KV Cache Performance Data

The framework expects CSV files for each KV cache configuration in `data/gov_report/`:

- `fullkv.csv`
- `snapkv_64.csv`
- `snapkv_128.csv`
- `snapkv_256.csv`
- `snapkv_512.csv`
- `snapkv_1024.csv`

Each file should contain:

```csv
times,energies,scores
2.1,1050.3,0.95
1.8,980.1,0.94
2.3,1100.2,0.96
...
```

- `times`: Processing time in seconds
- `energies`: Energy consumption in Joules
- `scores`: Quality score (0-1)


## Output Structure

After training and evaluation, the output directory contains:

```
results/
└── experiment_name/
    ├── models/
    │   ├── PPO/
    │   │   ├── PPO_final_1000000.zip
    │   │   ├── PPO_checkpoint_500000.zip
    │   │   └── best_model.zip
    │   └── DQN/
    │       ├── DQN_final_1000000.zip
    │       └── best_model.zip
    ├── logs/
    │   ├── PPO/
    │   └── DQN/
    ├── plots/
    │   ├── action_timeline_20240115_143022.png
    │   ├── performance_comparison_20240115_143022.png
    │   └── training_curves_20240115_143022.png
    └── training_summary.txt
```

## Monitoring and Metrics

### Key Metrics Tracked

- **Success Rate**: Percentage of successfully processed requests
- **Denial Rate**: Percentage of denied requests (if deny action enabled)
- **Average Score**: Quality score of processed requests
- **Energy Cost**: Total energy consumption cost
- **Latency**: Average request processing latency
- **Profit**: Revenue minus energy costs

[//]: # (## License)

[//]: # ()
[//]: # (This project is licensed under the MIT License - see the LICENSE file for details.)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this framework in your research, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@software{llm_datacenter_rl,)

[//]: # (  title={LLM Data Center RL: Reinforcement Learning for KV-cache Optimization},)

[//]: # (  author={LLM Data Center Research Team},)

[//]: # (  year={2024},)

[//]: # (  url={https://github.com/username/llm-datacenter-rl})

[//]: # (})

[//]: # (```)