import os, argparse, glob
from typing import List
from gymnasium.wrappers import RecordVideo
from envs.cartpole_variants import make_env
from stable_baselines3 import PPO

# optional concat
def maybe_concat(mp4_paths: List[str], out_path: str):
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except Exception as e:
        print(f"[concat] moviepy not available: {e}")
        return
    if not mp4_paths:
        return
    clips = [VideoFileClip(p) for p in mp4_paths]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(out_path, codec="libx264", audio=False)
    for c in clips:
        c.close()
    print(f"[concat] wrote {out_path}")

def run_record(model_path: str, variant: str, episodes: int, outdir: str, max_steps: int, render_fps: int, concat: bool):
    os.makedirs(outdir, exist_ok=True)
    # ensure video-friendly render + slower FPS so failures are clear
    env = make_env(variant, render_mode="rgb_array")
    try:
        env.unwrapped.metadata["render_fps"] = render_fps
    except Exception:
        pass

    env = RecordVideo(
        env,
        video_folder=outdir,
        name_prefix=f"{os.path.basename(model_path)}_{variant}",
    )
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        # after each episode, RecordVideo auto-saves the file

    env.close()

    # optional: stitch all episode files into one long clip
    if concat:
        pattern = os.path.join(outdir, f"{os.path.basename(model_path)}_{variant}-episode-*.mp4")
        parts = sorted(glob.glob(pattern))
        if parts:
            out_long = os.path.join(outdir, f"{os.path.basename(model_path)}_{variant}_LONG.mp4")
            maybe_concat(parts, out_long)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=str, default="runs/ppo_cart_ab")
    p.add_argument("--after",  type=str, default="runs/ppo_cart_unlearnA")
    p.add_argument("--episodes", type=int, default=5, help="number of episodes per case")
    p.add_argument("--max_steps", type=int, default=600, help="cap per-episode steps (default CartPole is 500)")
    p.add_argument("--render_fps", type=int, default=20, help="lower FPS to make motion clearer")
    p.add_argument("--outdir", type=str, default="videos")
    p.add_argument("--concat", action="store_true", help="stitch episodes into a single LONG.mp4 per case")
    args = p.parse_args()

    # Before: A and B
    run_record(args.before, "A", args.episodes, os.path.join(args.outdir, "before", "A"), args.max_steps, args.render_fps, args.concat)
    run_record(args.before, "B", args.episodes, os.path.join(args.outdir, "before", "B"), args.max_steps, args.render_fps, args.concat)
    # After: A and B
    run_record(args.after,  "A", args.episodes, os.path.join(args.outdir, "after",  "A"), args.max_steps, args.render_fps, args.concat)
    run_record(args.after,  "B", args.episodes, os.path.join(args.outdir, "after",  "B"), args.max_steps, args.render_fps, args.concat)

    print(f"🎥 Videos saved under {args.outdir}\\before\\(A|B) and {args.outdir}\\after\\(A|B)")

if __name__ == "__main__":
    main()
