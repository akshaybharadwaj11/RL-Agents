# Experimental Setup and Results

## Overview

This document details the experimental configurations, training procedures, and results for the Deep Q-Learning project on AirRaid-v5.

## Experimental Configurations

### 1. Baseline Configuration

**Purpose**: Establish performance baseline using standard DQN parameters

**Hyperparameters**:
```python
{
    'learning_rate': 0.00025,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay_steps': 500000,
    'batch_size': 32,
    'replay_buffer_size': 50000,
    'target_update_frequency': 5000,
    'n_frames': 4,
    'exploration_policy': 'epsilon_greedy'
}
```

**Results**:
- Average Test Reward: 53.0
- Average Episode Length: 1640.48
- Training Reward (last 50): 363.0

**Analysis**:
- Moderate performance with high exploration throughout training
- Long episodes suggest inefficient strategies
- High training/test gap indicates overfitting to exploration

### 2. Boltzmann Agent (Softmax Exploration)

**Purpose**: Test informed exploration strategy

**Modifications**:
- Changed exploration from ε-greedy to Boltzmann (softmax)
- Temperature parameter: start=1.0, end=0.1, decay=0.995

**Results**:
- Average Test Reward: 104.25 (+97% vs baseline)
- Average Episode Length: 2588.72
- Training Reward (last 50): 630.0

**Analysis**:
- Significant improvement over uniform random exploration
- Longer episodes but higher rewards
- Weighted exploration improved sample efficiency

### 3. Fast Decay

**Purpose**: Test accelerated transition to exploitation

**Modifications**:
- epsilon_decay_steps: 100000 (5× faster)
- Reaches epsilon_end much earlier in training

**Results**:
- Average Test Reward: 213.5 (+303% vs baseline)
- Average Episode Length: 172.15 (most efficient)
- Training Reward (last 50): 307.0

**Analysis**:
- Best test performance among exploration variants
- Short episodes indicate efficient, direct strategies
- Environment characteristics favored early exploitation

### 4. High Alpha (Learning Rate)

**Purpose**: Test effect of faster learning

**Modifications**:
- learning_rate: 0.001 (4× baseline)

**Results**:
- Average Test Reward: 274.0 (+417% vs baseline) ⭐ BEST
- Average Episode Length: 178.38
- Training Reward (last 50): 326.5

**Analysis**:
- Best overall performance
- Faster convergence to optimal policy
- Demonstrates that faster learning was beneficial for this environment
- No instability issues despite higher learning rate

### 5. Low Gamma (Discount Factor)

**Purpose**: Test importance of long-term planning

**Modifications**:
- gamma: 0.9 (vs 0.99 baseline)

**Results**:
- Average Test Reward: 95.75 (-48% vs high_alpha)
- Average Episode Length: 2330.99 (very long)
- Training Reward (last 50): 280.5

**Analysis**:
- Poor performance confirms need for long-term planning
- Myopic behavior leads to wandering
- AirRaid requires planning horizon of 100+ steps

## Training Details

### Environment Preprocessing

1. **Frame Processing**:
   - Convert RGB (210×160×3) to grayscale
   - Resize to 84×84
   - Normalize to [0, 1]

2. **Frame Stacking**:
   - Stack 4 consecutive frames
   - Provides temporal information (velocity, motion)

3. **Reward Clipping**:
   - Clip to {-1, 0, +1} for stability

### Training Procedure

**Hardware**:
- GPU: NVIDIA GPU with CUDA support
- CPU: Multi-core processor
- RAM: 16GB+

**Training Time**:
- Episodes: 5000
- Approximate time: 6-12 hours per configuration
- Total steps: ~1-2 million

**Checkpointing**:
- Save model every 100 episodes
- Keep best model based on average reward

## Performance Metrics

1. **Average Test Reward**: Mean reward over test episodes (no exploration)
2. **Average Episode Length**: Mean number of steps per episode
3. **Training Reward (last 50)**: Average reward during final training phase


## Key Findings

### 1. Learning Rate Impact
- **Finding**: Higher learning rate (4× baseline) achieved best performance
- **Explanation**: Environment allows faster learning without instability
- **Recommendation**: Start with higher learning rates for similar problems

### 2. Exploration Strategy
- **Finding**: Boltzmann exploration outperformed ε-greedy
- **Explanation**: Weighted exploration based on Q-values more efficient
- **Recommendation**: Consider Boltzmann for discrete action spaces

### 3. Exploration Schedule
- **Finding**: Fast decay to exploitation improved performance
- **Explanation**: Environment has relatively straightforward optimal strategy
- **Recommendation**: Tune decay schedule to environment complexity

### 4. Discount Factor Importance
- **Finding**: Low gamma significantly degraded performance
- **Explanation**: AirRaid requires long-term strategic planning
- **Recommendation**: Use γ ≥ 0.99 for games requiring multi-step reasoning

### 5. Efficiency vs. Reward
- **Finding**: Best performers had shortest episode lengths
- **Explanation**: Efficient strategies reach goals quickly
- **Recommendation**: Track episode length as quality indicator

## Reproducibility

### Random Seeds
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Environment Version
- Gymnasium: 0.29.0
- ALE-py: 0.8.1
- ROM: AirRaid (Atari 2600)

### Hardware Considerations
- GPU >= 16GB
- CPU >=16GB

## Future Experiments

### Planned
1. **Double DQN**: Address overestimation bias
2. **Dueling DQN**: Separate value and advantage streams
3. **Prioritized Replay**: Sample important transitions more frequently
4. **Rainbow**: Combine multiple improvements
5. **Multi-game**: Test generalization across Atari games

### Hyperparameter Tuning
- Systematic grid search
- Bayesian optimization
- Population-based training

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. AAAI, 2094-2100.
3. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML, 1995-2003.
"""