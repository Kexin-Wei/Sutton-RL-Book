import gym
all_envs = gym.envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
check_env='FrozenLake-v0'
if check_env in env_ids:
    print(1)
else:
    for env in env_ids:
        print(env)
