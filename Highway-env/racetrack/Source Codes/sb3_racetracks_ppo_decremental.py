import os
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import highway_env  # registers racetrack environments

# Optional system monitoring libs
try:
    import psutil
except Exception:
    psutil = None

try:
    import GPUtil
except Exception:
    GPUtil = None

# CONFIG
TRAIN = True
UNLEARN = True
N_CPU = 2  # reduce number of subprocess workers for stability
BATCH_SIZE = 64
MODEL_DIR = "racetrack_ppo"
TB_DIR = os.path.join(MODEL_DIR, "tb")  # primary TensorBoard log (optional)
BENCHMARK_EXCEL = os.path.join(MODEL_DIR, "benchmark.xlsx")
SAMPLE_EVERY_STEPS = 512  # how often the callback samples system metrics during training / unlearning

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)


def safe_make_vec_env(env_id: str, n_envs: int):
    """
    Create a vectorized environment, falling back to DummyVecEnv if subprocess creation fails.
    """
    try:
        return make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    except Exception as e:
        print("Warning: SubprocVecEnv failed, falling back to DummyVecEnv:", e)
        return make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)


#System sampling helpers
def sample_system_metrics():
    """
    Return a dict with timestamp, cpu_percent, mem_percent, gpu_percent (if available), and load averages.
    If psutil or GPUtil is not installed, values will be None.
    """
    now = time.time()
    cpu = None
    mem = None
    gpu = None
    load = None
    try:
        if psutil:
            # psutil.cpu_percent() gives instantaneous since last call; use short interval for somewhat accurate reading
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            try:
                load = os.getloadavg()  # returns (1,5,15) minute load averages on Unix
            except Exception:
                load = None
    except Exception:
        cpu = None
        mem = None
    try:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if len(gpus) > 0:
                # average GPU load across all GPUs (0-1 float)
                gpu = float(np.mean([g.load * 100.0 for g in gpus]))
    except Exception:
        gpu = None

    return {"timestamp": now, "cpu_percent": cpu, "mem_percent": mem, "gpu_percent": gpu, "loadavg": load}


#Training callback to sample system metrics
class BenchmarkCallback(BaseCallback):
    """
    Samples system metrics every `sample_every_steps` steps and stores them in self.system_samples.
    Also records per-callback-phase summary metrics when `on_training_start` and `on_training_end` are called externally.
    """

    def __init__(self, sample_every_steps: int = 512, verbose: int = 0):
        super().__init__(verbose)
        self.sample_every_steps = max(1, int(sample_every_steps))
        self.system_samples = []  # list of dicts {phase, step, num_timesteps, ...system metrics...}
        self._last_sample_step = 0

    def _on_step(self) -> bool:
        # Called at every environment step during learning
        n = self.num_timesteps
        if (n - self._last_sample_step) >= self.sample_every_steps:
            metrics = sample_system_metrics()
            metrics.update({"phase": getattr(self, "phase", "unknown"), "step": n, "timesteps": n})
            self.system_samples.append(metrics)
            self._last_sample_step = n
        return True

    # Utility to attach a phase label that will be used in saved samples.
    def set_phase(self, phase_name: str):
        self.phase = phase_name

    # Convenience to export collected samples to pandas DataFrame
    def to_dataframe(self):
        if len(self.system_samples) == 0:
            return pd.DataFrame()
        # Flatten loadavg (if present) to strings to store in DataFrame
        records = []
        for s in self.system_samples:
            rec = dict(s)
            if rec.get("loadavg") is not None:
                try:
                    rec["loadavg_1m"] = rec["loadavg"][0]
                    rec["loadavg_5m"] = rec["loadavg"][1]
                    rec["loadavg_15m"] = rec["loadavg"][2]
                except Exception:
                    rec["loadavg_1m"] = rec["loadavg"]
                rec.pop("loadavg", None)
            else:
                rec["loadavg_1m"] = None
                rec["loadavg_5m"] = None
                rec["loadavg_15m"] = None
            records.append(rec)
        return pd.DataFrame(records)


#Utility to run evaluation episodes and capture per-episode metrics
def run_evaluation_episodes(model, env, n_episodes=5, deterministic=True):
    """
    Runs `n_episodes` episodes in `env` with the provided `model`.
    Returns a list of dicts: {episode_idx, reward, length, start_time, end_time, duration, avg_cpu, avg_mem, avg_gpu}
    Note: env should be a single gym.Env (not vectorized) and follow Gymnasium's reset/step signatures.
    """
    episode_records = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        samples = []
        start_ts = time.time()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            # sample system metrics each step (cheap)
            samples.append(sample_system_metrics())
        end_ts = time.time()
        # compute averages
        cpu_vals = [s["cpu_percent"] for s in samples if s["cpu_percent"] is not None]
        mem_vals = [s["mem_percent"] for s in samples if s["mem_percent"] is not None]
        gpu_vals = [s["gpu_percent"] for s in samples if s["gpu_percent"] is not None]
        rec = {
            "episode_idx": ep,
            "reward": ep_reward,
            "length": ep_len,
            "start_time": start_ts,
            "end_time": end_ts,
            "duration_s": end_ts - start_ts,
            "avg_cpu_percent": float(np.mean(cpu_vals)) if len(cpu_vals) > 0 else None,
            "max_cpu_percent": float(np.max(cpu_vals)) if len(cpu_vals) > 0 else None,
            "avg_mem_percent": float(np.mean(mem_vals)) if len(mem_vals) > 0 else None,
            "avg_gpu_percent": float(np.mean(gpu_vals)) if len(gpu_vals) > 0 else None,
        }
        episode_records.append(rec)
    return episode_records

if __name__ == "__main__":
    # Limit PyTorch threads for macOS stability
    torch.set_num_threads(1)

    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in this process
        pass

    # Create vectorized environment safely for training
    env = safe_make_vec_env("racetrack-v0", n_envs=N_CPU)

    # Initialize PPO
    n_steps = max(1, BATCH_SIZE * 12 // max(1, N_CPU))  # ensure integer >= 1
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=n_steps,
        batch_size=BATCH_SIZE,
        n_epochs=5,
        learning_rate=1e-3,
        gamma=0.9,
        verbose=1,
        tensorboard_log=TB_DIR,
    )

    # create callback used to sample system metrics during training/unlearning
    bench_cb = BenchmarkCallback(sample_every_steps=SAMPLE_EVERY_STEPS, verbose=0)

    training_summary = []
    unlearning_summary = []
    eval_records = []

    #PHASE 1: TRAINING
    if TRAIN:
        print("Training the model to race properly...")
        bench_cb.set_phase("training")
        t0 = time.time()
        model.learn(total_timesteps=int(2e4), callback=bench_cb)
        t1 = time.time()
        training_summary.append(
            {
                "phase": "training",
                "timesteps": int(2e4),
                "start_time": t0,
                "end_time": t1,
                "duration_s": t1 - t0,
                "timesteps_per_second": int(2e4) / max(1e-6, (t1 - t0)),
            }
        )
        model.save(os.path.join(MODEL_DIR, "model"))
        del model

    # Reload the trained model with its environment
    model = PPO.load(os.path.join(MODEL_DIR, "model.zip"), env=env)

    #PHASE 2: DECREMENTAL UNLEARNING
    if UNLEARN:
        print("Starting decremental unlearning phase...")
        num_unlearning_stages = 5
        model.tensorboard_log = None  # disable TB logging during unlearning

        # Save unlearning-level system metrics separately
        for stage in range(num_unlearning_stages):
            print(f"Unlearning stage {stage + 1}/{num_unlearning_stages}")

            # Record system samples before modification
            bench_cb.set_phase(f"pre_unlearn_stage_{stage+1}")
            bench_cb._last_sample_step = model.num_timesteps  # sync step counter

            # Apply small decay + noise to policy weights (safe inside no_grad)
            with torch.no_grad():
                for param in model.policy.parameters():
                    param.mul_(0.98)
                    param.add_(0.02 * torch.randn_like(param))

            # Retrain briefly to stabilize the updated weights
            bench_cb.set_phase(f"unlearn_stage_{stage+1}_retrain")
            t0_stage = time.time()
            model.learn(
                total_timesteps=512,
                reset_num_timesteps=False,
                progress_bar=False,
                tb_log_name=None,
                callback=bench_cb,
            )
            t1_stage = time.time()
            unlearning_summary.append(
                {
                    "stage": stage + 1,
                    "timesteps": 512,
                    "start_time": t0_stage,
                    "end_time": t1_stage,
                    "duration_s": t1_stage - t0_stage,
                    "timesteps_per_second": 512.0 / max(1e-6, (t1_stage - t0_stage)),
                }
            )

        model.save(os.path.join(MODEL_DIR, "model_unlearned"))
        print("Unlearning complete, model saved.")

    #PHASE 3: RECORD VIDEOS & EVAL EPISODES
    # We'll create a fresh render environment (non-vectorized) for evaluation/video runs
    render_env = gym.make("racetrack-v0", render_mode="rgb_array")
    render_env = RecordVideo(
        render_env,
        video_folder=os.path.join(MODEL_DIR, "videos_unlearned"),
        episode_trigger=lambda e: True,
    )

    print("Recording decremental unlearning results and running evaluation episodes...")

    # Two video episodes (they will be saved by RecordVideo)
    # For each episode, we also gather per-episode system metrics
    ep_results = run_evaluation_episodes(model, render_env.env, n_episodes=2, deterministic=True)
    # run_evaluation_episodes used render_env.env (inner env) because RecordVideo wraps it;
    # the actual video will be recorded by RecordVideo during normal stepping below too.

    # We still need to step through the RecordVideo-wrapped env to produce the actual videos
    # (RecordVideo writes files when the outer wrapper's reset/step are used).
    for video in range(2):
        done = truncated = False
        obs, info = render_env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = render_env.step(action)
            render_env.render()
    render_env.close()

    # Collect eval records
    eval_records.extend(ep_results)

    #EXPORT BENCHMARK DATA TO EXCEL
    print("Saving benchmark data to Excel...")

    # Build DataFrames
    df_system_samples = bench_cb.to_dataframe()
    df_training = pd.DataFrame(training_summary) if len(training_summary) > 0 else pd.DataFrame()
    df_unlearning = pd.DataFrame(unlearning_summary) if len(unlearning_summary) > 0 else pd.DataFrame()
    df_eval = pd.DataFrame(eval_records) if len(eval_records) > 0 else pd.DataFrame()

    # Add aggregated stats
    if not df_system_samples.empty:
        agg = df_system_samples.groupby("phase").agg(
            samples=("timestamp", "count"),
            mean_cpu=("cpu_percent", "mean"),
            max_cpu=("cpu_percent", "max"),
            mean_mem=("mem_percent", "mean"),
            mean_gpu=("gpu_percent", "mean"),
        ).reset_index()
    else:
        agg = pd.DataFrame()

    # Write multiple sheets
    with pd.ExcelWriter(BENCHMARK_EXCEL, engine="openpyxl") as writer:
        df_system_samples.to_excel(writer, sheet_name="system_samples", index=False)
        agg.to_excel(writer, sheet_name="system_summary_by_phase", index=False)
        df_training.to_excel(writer, sheet_name="training_summary", index=False)
        df_unlearning.to_excel(writer, sheet_name="unlearning_summary", index=False)
        df_eval.to_excel(writer, sheet_name="evaluation_episodes", index=False)

    print(f"Benchmark exported to: {BENCHMARK_EXCEL}")

    # close training env
    env.close()
