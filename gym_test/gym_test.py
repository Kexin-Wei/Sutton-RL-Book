import matplotlib.pyplot as plt
import numpy as np
import gym
class env_fl():
    def __init__(self):
        pass

    def render_fig(self,name):
        s0=int(self.observation)/4
        s1=int(self.observation)%4
        fig=plt.figure(figsize=(4,4))
        plt.xlim([0,4])
        plt.ylim([0,4])
        ax=fig.gca()
        ax.set_xticks(np.arange(0,5,1))
        ax.set_yticks(np.arange(0,5,1))
        plt.grid()  #set grid

        ax = fig.add_subplot(111)
        for T in self.Terminal:
            plt.gca().add_patch(plt.Rectangle((T[0],T[1]),1,1,color="black"))
        

        for i in range(4):
            for j in range(4):
                if self.Policy[i,j,0]!=0: plt.annotate(text="",xy=(i,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #left
                if self.Policy[i,j,1]!=0: plt.annotate(text="",xy=(i+0.5,j+1),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #up
                if self.Policy[i,j,2]!=0: plt.annotate(text="",xy=(i+1,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #right
                if self.Policy[i,j,3]!=0: plt.annotate(text="",xy=(i+0.5,j),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #down                          
        plt.savefig(name)
        plt.close()
        

def test1():
    import gym
    env = gym.make('FrozenLake-v0') # CartPole-v0,MountainCar-v0
    env.reset()
    for _ in range(1000):
        env.render()
        _,_,done,_=env.step(env.action_space.sample()) # take a random action
        if done:
            return
    env.close()

def test2():
    import gym
    env = gym.make('Copy-v0')
    env.reset()
    env.render()

def Atari():
    import gym
    env = gym.make('SpaceInvaders-v0') #SpaceInvaders-v0, MsPacman-v0
    env.reset()
    env.render()

def test3(): # check what is in observation
    import gym
    
    env = gym.make('FrozenLake-v0')
    end_steps=[]
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample() # take a random action
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                end_steps.append(t+1)
                break
    plt.plot(end_steps)
    plt.show()
    env.close()

def test4():
    import gym
    env = gym.make('CartPole-v0')
    highscore = 0
    point_record=[]
    for i_episode in range(20): # run 20 episodes
        observation = env.reset()
        points = 0 # keep track of the reward each episode
        while True: # run until episode is done
            env.render()
            action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left
            observation, reward, done, info = env.step(action)
            points += reward
            point_record.append(points)
            if done:
                break
    plt.plot(point_record)
    plt.show()
    env.close()           
        
    

if __name__=="__main__":
    test3()