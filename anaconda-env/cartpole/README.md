# README — RL Unlearning Project  
*Author: Nihindu Nethsika Hettiarachchi*  

---

## Overview
This project implements **selective unlearning for a Reinforcement Learning (RL) agent** using PPO from Stable-Baselines3.  
It trains a CartPole agent on two environment variants (A = hard/private, B = easy/public), then fine-tunes it to **forget A** while retaining performance on B.  
You can reproduce the full experiment (training → unlearning → evaluation → attack → video generation) using Anaconda.

---

## 🧩 Requirements and Setup

### Step 1. Create the environment
Open **Anaconda Prompt** and run:
```bash
conda create -n rl-unlearn python=3.10 -y
conda activate rl-unlearn
pip install stable-baselines3 gymnasium==0.28.1 numpy pandas matplotlib psutil moviepy imageio[ffmpeg] tensorboard imageio-ffmpeg
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 2. Go to the project folder
```bash
cd C:\Users\neths\rl-unlearning-project
```

### Step 3. Ensure `__init__.py` files exist
```bash
type NUL > envs\__init__.py
type NUL > src\__init__.py
type NUL > src\utils\__init__.py
```

---

## 🚀 Full Run Commands

Run these in order from Anaconda Prompt:

```bash
python -m src.train --steps 150000 --seed 42 --save runs/ppo_cart_ab
python -m src.unlearn --from_ckpt runs/ppo_cart_ab --finetune_steps 50000 --seed 123 --save runs/ppo_cart_unlearnA
python -m src.evaluate --before runs/ppo_cart_ab --after runs/ppo_cart_unlearnA --n_episodes 50
python -m src.attack --model runs/ppo_cart_ab --pop 80 --gens 80 --out results/attack_top.json
python -m src.record_videos --before runs/ppo_cart_ab --after runs/ppo_cart_unlearnA --episodes 3 --outdir videos
python -m src.benchmarks --eval_csv results/eval_full.csv --attack_csv results/attack_population.csv --out results/benchmarks.png
```

Optional quick test (short version):
```bash
python -m src.train --steps 30000 --save runs/ppo_cart_ab
python -m src.unlearn --from_ckpt runs/ppo_cart_ab --finetune_steps 10000 --save runs/ppo_cart_unlearnA
python -m src.evaluate --before runs/ppo_cart_ab --after runs/ppo_cart_unlearnA --n_episodes 10
```

---

## 📁 What Each File Does

| File | Description |
|------|--------------|
| `envs/cartpole_variants.py` | Defines CartPole A & B (A = harder, private; B = easier, public). |
| `src/train.py` | Trains PPO model on both environments, logs resource usage. |
| `src/unlearn.py` | Fine-tunes model on B only (simulates forgetting A). |
| `src/evaluate.py` | Evaluates both models before/after on both environments. |
| `src/attack.py` | Performs adversarial attack that maximizes Env A–B difference. |
| `src/record_videos.py` | Records MP4s of model behavior before/after on A/B. |
| `src/benchmarks.py` | Summarizes and plots evaluation and attack data. |
| `src/utils/monitor.py` | Logs CPU and memory usage to CSV during training. |
| `clean.bat` | Deletes runs/results/videos/tb folders for a fresh start. |

---

## 📊 Results Meaning

### `results/summary.csv`
Shows mean rewards for A/B before and after unlearning.  
✅ Unlearning works if A drops but B stays stable.

### `results/eval_full.csv`
Per-episode rewards — used for detailed analysis or plotting.

### `results/attack_top.json`
Shows the best adversarial action sequence and its fitness.  
High fitness = stronger behavioral difference between A and B.

### `results/attack_population.csv`
Evolution log of attack fitness per generation (for convergence checking).

### `results/usage_train.csv` and `results/usage_unlearn.csv`
CPU and memory utilization during training/unlearning — for benchmarking.

### `results/benchmarks.png`
Bar chart comparing before/after rewards and attack metrics.

### `videos/`
Contains recorded .mp4 gameplay for before/after, Env A/B.  
Visual proof that unlearning changed agent behavior.

---

## ✅ Interpreting Results

| Stage | Env A | Env B | Meaning |
|--------|--------|--------|----------|
| Before | High reward | High reward | Model knows both A and B |
| After | Low reward | High reward | Model forgot A but retained B |
| Attack | Strong A/B diff | Confirms selective forgetting |

---

## 🧠 Tips
- Start TensorBoard with:  
  `tensorboard --logdir tb --port 6006`
- View metrics live at [http://localhost:6006](http://localhost:6006)
- Use fewer steps for fast debugging runs.
- Each command can be rerun independently (use defaults).

---

## 🧩 Troubleshooting

### Missing moviepy
```bash
pip install moviepy imageio[ffmpeg]
```

### Torch DLL import error
```bash
pip uninstall -y torch torchvision torchaudio
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

### ModuleNotFoundError for src.utils
```bash
type NUL > src\__init__.py
type NUL > src\utils\__init__.py
type NUL > envs\__init__.py
```

### TensorBoard not showing anything
Make sure you started it before training and refresh the browser.

---

## 🏁 Conclusion
After running all scripts:
- You’ll have trained/unlearned PPO models (`runs/ppo_cart_ab`, `runs/ppo_cart_unlearnA`).
- Evaluation + attack + usage + videos saved under `/results` and `/videos`.
- The agent clearly forgets A while performing fine on B — proven by metrics and gameplay.

---

**Enjoy experimenting — this setup is plug-and-play for reproducible RL unlearning results.**
