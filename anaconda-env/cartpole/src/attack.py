import argparse, json, numpy as np, pandas as pd, os
from stable_baselines3 import PPO
from envs.cartpole_variants import make_env

def random_action_sequence(length):
    return list(np.random.randint(0, 2, size=length))

def mutate(seq, p=0.08):
    out = seq.copy()
    for i in range(len(out)):
        if np.random.rand() < p:
            out[i] = 1 - out[i]
    return out

def eval_sequence(model, env, seq):
    obs, _ = env.reset()
    total = 0.0
    for a in seq:
        obs, reward, terminated, truncated, _ = env.step(a)
        total += reward
        if terminated or truncated:
            break
    return total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="runs/ppo_cart_ab")   # no .zip
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--out", type=str, default="results/attack_top.json")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)

    model = PPO.load(args.model)
    envA, envB = make_env("A"), make_env("B")

    pop = [random_action_sequence(args.seq_len) for _ in range(args.pop)]
    records = []

    for gen in range(args.gens):
        scores = []
        for indiv in pop:
            scoreA = eval_sequence(model, envA, indiv)
            scoreB = eval_sequence(model, envB, indiv)
            fitness = abs(scoreA - scoreB)
            scores.append((fitness, scoreA, scoreB, indiv))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[0]
        records.append({
            "gen": gen,
            "fitness": float(top[0]),
            "scoreA": float(top[1]),
            "scoreB": float(top[2]),
        })

        keep = max(2, int(0.2 * len(pop)))
        new_pop = [s[3] for s in scores[:keep]]
        while len(new_pop) < args.pop:
            parent = new_pop[np.random.randint(0, len(new_pop))]
            new_pop.append(mutate(parent, p=0.12))
        pop = new_pop

    pd.DataFrame(records).to_csv("results/attack_population.csv", index=False)

    best = scores[0]
    result = {
        "fitness": float(best[0]),
        "scoreA": float(best[1]),
        "scoreB": float(best[2]),
        "sequence": [int(x) for x in best[3]],  # <-- convert NumPy ints to Python ints
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved attack top to {args.out} and population to results/attack_population.csv")

if __name__ == "__main__":
    main()
