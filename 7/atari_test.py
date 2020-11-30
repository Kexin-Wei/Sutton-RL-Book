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

def test2():
    n=4
    epsilon=0.2
    p=np.random.default_rng().dirichlet(np.ones(n))*(1-epsilon)
    p+=epsilon/n
    print(p)

if __name__ == "__main__":
    test2()


