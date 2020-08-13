from gym.envs.registration import register

register(
    id='ContinuousCartPole-v0',
    entry_point='envs.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=1000,
)