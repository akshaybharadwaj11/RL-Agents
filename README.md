# AirRaid-DQN
This repo demonstrates the implementation of Deep Q-Learning for playing Atari 2600's AirRaid game. The project trains a **Deep Q-Network (DQN) agent** using **Convolutional Neural Networks** with **experience replay** and **target networks**, achieving **417% improvement over baseline** through systematic hyperparameter optimization. It includes comprehensive experiments comparing **5 different configurations** (exploration strategies, learning rates, and discount factors) with detailed performance analysis and real-time visualization capabilities.

---

## Overview

| Component | Description |
|------------|-------------|
| **Environment** | [`ALE/AirRaid-v5`](https://ale.farama.org/environments/air_raid/) (Atari 2600) |
| **Algorithm** | Deep Q-Network (DQN) with Experience Replay |
| **Task** | Reinforcement Learning for Atari Game Playing |
| **Neural Network** | CNN with 3 Conv Layers + 2 FC Layers (512 hidden units) |
| **Input Processing** | 4 Stacked Grayscale Frames (84×84 pixels) |
| **Action Space** | Discrete(6): NOOP • FIRE • RIGHT • LEFT • RIGHTFIRE • LEFTFIRE |
| **Best Configuration** | High Alpha (Learning Rate = 0.001) → **274.0 Avg Reward** |
| **Evaluation Metrics** | Average Test Reward • Episode Length • Training Reward |
| **Frameworks** | PyTorch • Gymnasium • OpenCV • Matplotlib |
| **Hardware Requirements** | RAM >= 8GB • GPU >= 4GB (optional) • Disk Space >= 2GB |

---

## 📊 Results

### Performance Comparison

| Configuration | Avg Test Reward | Avg Episode Length | Training Reward (last 50) | Performance vs Baseline |
|--------------|-----------------|-------------------|--------------------------|------------------------|
| **baseline** | 53.0 | 1640.48 | 363.0 | Baseline |
| **boltzmann_agent** | 104.25 | 2588.72 | 630.0 | **+97%** 🟢 |
| **fast_decay** | 213.5 | 172.15 | 307.0 | **+303%** 🟢 |
| **high_alpha** | **274.0** | 178.38 | 326.5 | **+417%** 🏆 |
| **low_gamma** | 95.75 | 2330.99 | 280.5 | **+81%** 🟡 |

### Key Findings

✅ **High Learning Rate** achieved best performance (274.0 reward, 5.2× baseline)  
✅ **Fast Decay** produced most efficient policies (172.15 avg steps)  
✅ **Boltzmann Exploration** outperformed ε-greedy by 97%  
❌ **Low Gamma** demonstrated importance of long-term planning

---
## 🛠️ Technologies

### Core Frameworks
- **PyTorch 2.0+**: Deep learning
- **Gymnasium 0.29+**: RL environments
- **ALE-py 0.8+**: Atari emulator

### Libraries
- **NumPy**: Numerical computation
- **OpenCV**: Image processing
- **Matplotlib**: Visualization
- **tqdm**: Progress tracking

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/akshaybharadwaj11/RL-Agents.git
cd airraid-dqn
```

### 2. Create a Virtual Environment

```bash
conda create -n airraid-env python=3.10
conda activate airraid-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Atari ROMs

```bash
pip install gymnasium[atari,accept-rom-license]
```

---

## 🎯 Tasks

### Training

* **Baseline Training** - Standard DQN with ε-greedy exploration
* **Boltzmann Agent Training** - Softmax exploration strategy
* **Fast Decay Training** - Accelerated epsilon decay schedule
* **High Alpha Training** - Increased learning rate (0.001)
* **Low Gamma Training** - Reduced discount factor (0.9)

### Evaluation

* **Performance Analysis** - Comparative analysis of all configurations
* **Model Testing** - Test trained agents over multiple episodes

* [Training and Eval](Notebooks/airraid-rl-agent.ipynb)


### Inference & Visualization

* **[Inference & Gameplay Visualization](Notebooks/Inference_viz.ipynb)** - Real-time visualization with Q-values

---

## 📖 References

1. **Mnih, V., et al. (2015)**. Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533.
2. **Van Hasselt, H., et al. (2016)**. Deep Reinforcement Learning with Double Q-learning. *AAAI*.
3. **Gymnasium Documentation**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
4. **ALE Environment**: [https://ale.farama.org/environments/air_raid/](https://ale.farama.org/environments/air_raid/)

---

## 📧 Contact

**Akshay** - [akshaybharadwaj456@gmail.com](mailto:akshaybharadwaj456@gmail.com)

---


