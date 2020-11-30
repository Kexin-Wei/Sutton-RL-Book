import numpy as np
import gym

env=gym.make("Breakout-ram-v0")

env.reset()

def test():
    for _ in range(1000):
        env.render()
        _,_,done,_=env.step(env.action_space.sample()) # take a random action
        if done:
            return
    env.close()

if __name__ == "__main__":
    test()


