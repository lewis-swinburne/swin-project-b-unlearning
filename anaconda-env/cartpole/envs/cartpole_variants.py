import gymnasium as gym

def make_env(variant: str, render_mode: str | None = None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    base = env.unwrapped

    if variant.upper() == "A":
        base.gravity = 34.0
        base.length = 0.16
        base.force_mag = 2.5
        base.theta_threshold_radians = 0.06
        base.x_threshold = 0.9
        env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=150)
    else:  # variant B
        base.gravity = 9.0
        base.length = 0.7
        base.force_mag = 12.0
        base.theta_threshold_radians = 0.28
        base.x_threshold = 2.8

    base.polemass_length = base.masspole * base.length
    base.total_mass = base.masspole + base.masscart

    env.variant = variant.upper()
    return env
