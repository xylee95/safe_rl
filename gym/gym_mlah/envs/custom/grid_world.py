import numpy as np
import gym
from gym import spaces
import os

NOTHING = 0
LEFT = 1
RIGHT = 2
DOWN = 3
UP = 4

GOAL_REWARD = 100.0
STEP_REWARD = -1.0

class GridWorld(gym.Env):

    def __init__(self,n_dims=21,goal=[11,11],seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.dim = n_dims
        self.set_goal(goal)
        self.action_space = spaces.Discrete(5)
        self.low = np.array([0, 0])
        self.high = np.array([self.dim-1, self.dim-1])
        self.observation_space =  spaces.Box(self.low, self.high)
        self.state = self.reset()
        self.grid = []
        self.success_count = 0 #to use as ID for filenme
        self.traj_counter = 1 #keep track of new trajectories
        self.total_step_count = 0 #keep track when a new trajectory begins
        self.experiment_id = 0

    def set_experiment_id(self,id):
        self.experiment_id = id

    def step(self,action):
        self.total_step_count += 1
        if self.total_step_count%1024 == 0 :
            #print('Checkpoint: New Trajectory')
            self.traj_counter +=1
            #uncomment to save coordinates
            #if self.traj_counter > 40:
            #    os.makedirs('render_frames/' + str(self.experiment_id) + '/'+ str(self.traj_counter))  

        if(action == NOTHING):
            row = self.state[0]
            col = self.state[1]

        elif(action == LEFT):
            row = self.state[0]
            col = max(self.state[1] - 1, 0)

        elif(action == RIGHT):
            row = min(self.state[0] + 1, self.dim-1)
            col = self.state[1]

        elif(action == DOWN):
            row = self.state[0]
            col = min(self.state[1] + 1, self.dim-1)

        elif(action == UP):
            row = max(self.state[0] - 1, 0)
            col = self.state[1]

        state_ = np.array([row,col],dtype=int)


        if np.array_equal(state_, self.goal):
            reward = GOAL_REWARD
            self.done = True
            self.success_count+=1
            #uncomment to save coordinates
            #self.saveXY(self.success_count, self.traj_counter)
        else:
            reward = STEP_REWARD + 10*(-self.l2_dist(state_)+self.l2_dist(self.state))
        
        self.state = state_
        return np.array(self.state,dtype=np.float64), reward, self.done, {}

    def set_goal(self,goal):
        self.goal = np.array(goal, dtype=int)

    def reset(self):
        row,col = self.grid_select()
        self.state = np.array([row,col],dtype=int)
        self.done = False
        self.grid = []
        return self.state

    def l2_dist(self,state):
        inner = (self.goal[0]-state[0])**2 + (self.goal[1]-state[1])**2
        return np.sqrt(inner)

    def grid_select(self):
        row = np.random.randint(0,self.dim)
        col = np.random.randint(0,self.dim)
        return row,col

    def render_xy(self, render_attack, fake_obs, mode='human'):
        if render_attack==0:
            line = str(self.state[0]) + ' ' + str(self.state[1]) + ' '+ str(self.state[0]) + ' ' + str(self.state[1]) + str(' 0')
            self.grid.append(line)
        else:
            line = str(self.state[0]) + ' '+ str(self.state[1]) + ' '+ str(fake_obs[0]) + ' ' + str(fake_obs[1]) + str(' 1')
            self.grid.append(line)

    def render(self, render_attack, fake_obs, mode='human'):
        if render_attack==0:
            image = []
            for i in range(self.dim):
                line = ' '
                for j in range(self.dim):
                    if(np.array_equal(self.goal,np.array([i,j],dtype=int))):
                        pol_symb = '1'
                    elif(np.array_equal(self.state,np.array([i,j],dtype=int))):
                        pol_symb = '0.5'
                    else:
                        pol_symb = '0'

                    line += pol_symb + ' '
                image.append(line)
        else:
            image = []
            for i in range(self.dim):
                line = ' '
                for j in range(self.dim):
                    #goal
                    if(np.array_equal(self.goal,np.array([i,j],dtype=int))):
                        pol_symb = '1'

                    #agent/state
                    elif(np.array_equal(self.state,np.array([i,j],dtype=int))):
                        pol_symb = '0.5'

                    #fake observation
                    elif(np.array_equal(fake_obs,np.array([i,j],dtype=int))):
                       # print('True state:', self.state)
                        #print('Fake observation: ',fake_obs)
                        pol_symb = '0.7'
                    #environment
                    else:
                        pol_symb = '0.2'

                    line += pol_symb + ' '
                image.append(line)
        
        #print(image)
        self.grid.append(image)

    def saveXY(self,success_count,trajectory):
        i = success_count
        if trajectory > 40:
            with open('render_frames/' + str(self.experiment_id) + '/' + str(trajectory) + '/printtxt_linear_%i.txt'%i, 'w') as f:
                for item in self.grid:
                        f.write(item)
                        f.write('\n')

    def saveGrid(self,success_count,trajectory):
        i = success_count
        if trajectory > 40:
            with open('render_frames/' + str(trajectory) + '/printtxt_mirror_%i.txt'%i, 'w') as f:
                for item in self.grid:
                    for line in item:
                        f.write(line)
                        f.write('\n')

            #print('Steps Taken to Goal',len(self.grid))
        #else:
         #   print('Not saving for warmup runs')

    def render_cmd(self, mode='human'):
        print('GridWorld')
        bar = ''
        for i in range(self.dim):
            bar += '----'
        bar += '-'
        print(bar)
        for i in range(self.dim):
            line = '|'
            for j in range(self.dim):
                if(np.array_equal(self.goal,np.array([i,j],dtype=int))):
                    pol_symb = 'G'
                elif(np.array_equal(self.state,np.array([i,j],dtype=int))):
                    pol_symb = 'A'
                else:
                    pol_symb = ' '

                line += ' %s |' % pol_symb
            print(line)
            print(bar)
        print('')

