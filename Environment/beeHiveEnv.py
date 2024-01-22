import copy
import logging
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

import pettingzoo
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from enum import Enum

logger = logging.getLogger(__name__)

SEASONS = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

CELL_SIZE = 30

BACKGROUND_COLOR = (222, 237, 147)
LETTERS_COLOR = 'black'
HIVE_COLOR = 'orange'
FLOWER_COLOR = 'deeppink'
PESTICIDE_COLOR = 'indigo'

# fixed colors for #agents = n_max <= 10
AGENTS_COLOR = "yellow"

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A',
    'hive': 'H',
    'flower': 'F',
    'pesticide': 'P'
}

class BeeHiveEnv(gym.Env, AECEnv):
    def __init__(self,
                 n_agents,
                 grid_shape,
                 max_steps=364,
                 season=SEASONS[0],
                 num_flowers=40,
                 rain=False
                 ):

        self.seed()

        # number of agents in the colony
        self.n_agents = n_agents
        self._grid_shape = grid_shape

        # elements
        self.season = season
        self.num_honey = 0
        self.initial_flowers = num_flowers
        self.num_flowers = num_flowers
        self.num_pesticides = int(0.3 * num_flowers)
        self.rain = rain
        self._agent_view_mask = (30, 30)
        
        # the list of agents
        self.agents = [str(i) for i in range(n_agents)]
        self.agent_locations = {agent: (0, 0) for agent in self.agents}

        # (0,0) is the colony
        self.hive = (0, 0)
        
        self.flower_locations = []
        self.pesticides_locations = []

        # number of steps taken on the current day
        self.steps = 0
        self.total_turns = 0
        self.max_steps = max_steps
        self.n_actions = 5

        self.initialize_positions()

        # the possible actions an agent can take
        self.action_space = MultiAgentActionSpace([spaces.Discrete(4) for _ in self.agents])

        self._obs_high = np.array([self._grid_shape[0], self._grid_shape[1]] + [4] * 8)
        self._obs_low = np.array([0] * 10)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in self.agents])

        self.done = False  # hive is alive
        self.dones = {i: False for i in self.agents} #agents are alive
        self.infos = {agent: {} for agent in self.agents}
        self.viewer = None

        self._base_img = self.__draw_base_img()

    def reset(self):
        self.season = SEASONS[0]
        self.rain = False
        self.total_turns = 0
        self.rewards = {i: 0 for i in self.agents}
        self.done = False
        self.dones = {i: False for i in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # the amount of individual energy each agent has (in the bee hive it represents how many flowers it can go to)
        # agents start with 3
        self.agent_energies = {agent: 3 for agent in self.agents}
        self.agent_pollination = {agent: 0 for agent in self.agents}
        self.steps = 0
        self.agent_locations = {agent: (0, 0) for agent in self.agents}
        self.__draw_base_img()
        self.num_flowers = self.initial_flowers
        self.flower_locations = []
        self.pesticides_locations = []
        self.initialize_positions()
        self.num_honey = 0
        # set the agent iterator
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return [self.simplified_features() for _ in range(self.n_agents)]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def step(self, action):
        
        self.steps = self.total_turns // self.n_agents
        a = action
        self._cumulative_rewards[self.agent_selection] = 0
        self.rewards[self.agent_selection] = 0
        self.make_more_bees()
        self.make_more_flowers()
        self.make_pesticides()
        
        # only perform an action if the agent is alive
        if not self.dones[self.agent_selection]:

            # check which action was taken
            if a == 0:
                self.move_left()
            elif a == 1:
                self.move_right()
            elif a == 2:
                self.move_up()
            elif a == 3:
                self.move_down()
            # elif a == 4:
            #   self.stay()


        # if all agents die, the sim is over
        if self.num_alive_agents() == 0 or (self.num_honey <= 0 and self.season == SEASONS[3]):
            self.done = True
            self.close()

        # selects the next agent.
        if self.num_alive_agents() != 0:
            self.agent_selection = self._agent_selector.next()
        else:
            self.done = True
            #self.close()
            
        if (self.steps >= self.max_steps):
            self.done = True
            for i in range(self.n_agents):
                self.dones[i] = True
        
        self.total_turns += 1
        self._accumulate_rewards()

        return [self.simplified_features(), self.done]

    def render(self, mode='human'):
        
        if 0 <= self.steps <= 91 :
            self.season = SEASONS[0]
        elif 91 < self.steps <= 182 :
            self.season = SEASONS[1]
        elif 182 < self.steps <= 273 :
            self.season = SEASONS[2]
        elif 273  < self.steps <= 364:
            self.season = SEASONS[3]
        else:
            self.close()
            
        self.print_info()

        img = copy.copy(self._base_img)

        for flower_pos in self.flower_locations:
            fill_cell(img, flower_pos, cell_size=CELL_SIZE, fill=FLOWER_COLOR)

        for pesticide_pos in self.pesticides_locations:
            fill_cell(img, pesticide_pos, cell_size=CELL_SIZE, fill=PESTICIDE_COLOR)

        for agent in self.agents:
            if not self.dones[agent]:
                fill_cell(img, self.agent_locations[agent], cell_size=CELL_SIZE, fill=AGENTS_COLOR)
                write_cell_text(img, text="Bee", pos=self.agent_locations[agent], cell_size=CELL_SIZE, fill=LETTERS_COLOR,
                                margin=0.3)

        fill_cell(img, (0, 0), cell_size=CELL_SIZE, fill=HIVE_COLOR)
        write_cell_text(img, text="Hive", pos=(0, 0), cell_size=CELL_SIZE, fill=LETTERS_COLOR, margin=0.15)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.num_alive_agents() > 0 and self.num_honey > 0:
            print("HIVE SURVIVED! :)")
            
        else:
            print("HIVE DIED :(")
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            #exit()
            #exit()

    def seed(self, seed=None):
        print(seed)
        randomizer, s = seeding.np_random(seed)
        self.randomizer = randomizer


    def __draw_base_img(self):
        # create grid and make everything black
        img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=BACKGROUND_COLOR)

        # draws background
        for i in range(self._grid_shape[0]):
            for j in range(self._grid_shape[1]):
                fill_cell(img, (i, j), cell_size=CELL_SIZE, fill=BACKGROUND_COLOR, margin=0.02)

        return img

    def initialize_positions(self):
        
        x = self._grid_shape[0]
        y = self._grid_shape[1]

        self.pesticides_locations = []
        self.flower_locations = []

        # set flowers
        for i in range(self.num_flowers):

            x_rand = self.randomizer.randint(0, x)
            y_rand = self.randomizer.randint(0, y)

            while (x_rand, y_rand) == (0, 0) or (x_rand, y_rand) in self.flower_locations:
                x_rand = self.randomizer.randint(0, x)
                y_rand = self.randomizer.randint(0, y)
            self.flower_locations.append((x_rand, y_rand))

        # set traps (flowers with pesticides)
        for i in range(self.num_pesticides):
            x_rand = self.randomizer.randint(0, x)
            y_rand = self.randomizer.randint(0, y)
            while (x_rand, y_rand) == (0, 0) or (x_rand, y_rand) in self.flower_locations or (x_rand, y_rand) in self.pesticides_locations:
                x_rand = self.randomizer.randint(0, x)
                y_rand = self.randomizer.randint(0, y)
            self.pesticides_locations.append((x_rand, y_rand))

    def pollinate(self, pos):
        
        (x, y) = pos
        if self.agent_pollination[self.agent_selection] > 0 and self.agent_energies[self.agent_selection] <= 0 and (x, y) not in self.pesticides_locations and (x, y) not in self.flower_locations:
            
            make = random.randint(0, 4)
            if make == 1:
                self.flower_locations.append((x, y))
                self.agent_pollination[self.agent_selection] -= 1
                self.num_flowers += 1

    def make_pesticides(self):
        
        if 0.25 * self.num_flowers > self.num_pesticides:   
            
            i = random.randint(0, len(self.flower_locations) - 1)  
            if self.flower_locations[i]:
                (x, y) = self.flower_locations[i]
                self.flower_locations.remove((x, y))
                self.pesticides_locations.append((x, y))
                self.num_pesticides += 1
                self.num_flowers -= 1
    
    def make_more_bees(self):
    
        if self.num_honey >= 1 and self.num_alive_agents() < self.n_agents:

            agent = self.agent_selection
            if self.agent_locations[agent] == (-1, -1):
                self.agent_locations[agent] = (0, 0)
                self.dones[agent] = False
                self.infos[agent] = {}
                self.agent_energies[agent] = 3
                self.agent_pollination[agent] = 0
                self.rewards[agent] = 0
                self._cumulative_rewards[agent] = 0
                self.num_honey -= 1
                
    def make_more_flowers(self):
        if self.season == SEASONS[0]:
            x_grid = self._grid_shape[0]
            y_grid = self._grid_shape[1]

            prob = random.randint(0, 20)
            if prob == 1:
                x_rand = self.randomizer.randint(0, x_grid)
                y_rand = self.randomizer.randint(0, y_grid)

                while (x_rand, y_rand) == (0, 0) or (x_rand, y_rand) in self.flower_locations or (
                        x_rand, y_rand) in self.pesticides_locations:
                    x_rand = self.randomizer.randint(0, x_grid)
                    y_rand = self.randomizer.randint(0, y_grid)

                self.flower_locations.append((x_rand, y_rand))
                self.num_flowers += 1

        elif self.season == SEASONS[1]:
            x_grid = self._grid_shape[0]
            y_grid = self._grid_shape[1]

            prob = random.randint(0, 45)
            if prob == 1:
                x_rand = self.randomizer.randint(0, x_grid)
                y_rand = self.randomizer.randint(0, y_grid)

                while (x_rand, y_rand) == (0, 0) or (x_rand, y_rand) in self.flower_locations or (
                        x_rand, y_rand) in self.pesticides_locations:
                    x_rand = self.randomizer.randint(0, x_grid)
                    y_rand = self.randomizer.randint(0, y_grid)

                self.flower_locations.append((x_rand, y_rand))
                self.num_flowers += 1
                
        elif self.season == SEASONS[2]:
            
            # pollination stopped
            self.agent_pollination = {agent: 0 for agent in self.agents}
            
            x_grid = self._grid_shape[0]
            y_grid = self._grid_shape[1]

            prob = random.randint(0, 75)
            if prob == 1:
                x_rand = self.randomizer.randint(0, x_grid)
                y_rand = self.randomizer.randint(0, y_grid)

                while (x_rand, y_rand) == (0, 0) or (x_rand, y_rand) in self.flower_locations or (
                        x_rand, y_rand) in self.pesticides_locations:
                    x_rand = self.randomizer.randint(0, x_grid)
                    y_rand = self.randomizer.randint(0, y_grid)

                self.flower_locations.append((x_rand, y_rand))
                self.num_flowers += 1
                
        elif self.season == SEASONS[3]:
            
            self.agent_pollination = {agent: 0 for agent in self.agents}
            
            # bees start to consume honey (approx. 1 honey per day in the hive)
            prob = random.randint(0, self.n_agents)
            if prob == 1:
                self.num_honey -= 1
            
            # flowers start disappearing
            
            prob = random.randint(0, 75)
            if prob == 1:
                if self.num_flowers > 0:
                    i = random.randint(0, len(self.flower_locations) - 1)  
                    if self.flower_locations[i]:
                        (x, y) = self.flower_locations[i]
                        self.flower_locations.remove((x, y))
                        self.num_flowers -= 1
                
                if self.num_pesticides > 0:  
                    i = random.randint(0, len(self.pesticides_locations) - 1)  
                    if self.pesticides_locations[i]:
                        (x, y) = self.pesticides_locations[i]
                        self.pesticides_locations.remove((x, y))
                        self.num_pesticides -= 1
            
                           
    def move_left(self):
        agent = self.agent_selection
        (x, y) = self.agent_locations[agent]
        self.rewards[agent] -= 1
        
        if y > 0:
                            
            self.pollinate((x, y))
                
            # New Agent Location
            new_pos = (x, y - 1)
            self.agent_locations[agent] = new_pos
            
            self.check_in_hive(new_pos)
            self.pickup_pollen(new_pos)
            self.check_pesticide(new_pos)
        
        self.check_energy(agent)

    def move_right(self):
        agent = self.agent_selection
        (x, y) = self.agent_locations[agent]
        self.rewards[agent] -= 1
        
        if y < self._grid_shape[1] - 1:
            self.pollinate((x, y))
                
            # New Agent Location
            new_pos = (x, y + 1)
            self.agent_locations[agent] = new_pos
            
            self.check_in_hive(new_pos)
            self.pickup_pollen(new_pos)
            self.check_pesticide(new_pos)
            
        self.check_energy(agent)

    def move_down(self):

        agent = self.agent_selection
        (x, y) = self.agent_locations[agent]
        self.rewards[agent] -= 1
        
        if x < self._grid_shape[0] - 1:
            self.pollinate((x, y))
                
            # New Agent Location
            new_pos = (x + 1, y)
            self.agent_locations[agent] = new_pos
            
            self.check_in_hive(new_pos)
            self.pickup_pollen(new_pos)
            self.check_pesticide(new_pos)
            
        self.check_energy(agent)

    def move_up(self):
        agent = self.agent_selection
        (x, y) = self.agent_locations[agent]
        self.rewards[agent] -= 1
        
        if x > 0:
            self.pollinate((x, y))
            
            # New Agent Location
            new_pos = (x - 1, y)
            self.agent_locations[agent] = new_pos
            
            self.check_in_hive(new_pos)
            self.pickup_pollen(new_pos)
            self.check_pesticide(new_pos)
            
        self.check_energy(agent)

    # Handles picking up pollen from a flower and dropping off at the colony
    def pickup_pollen(self, pos):
        (x, y) = pos

        # Picking up pollen
        if (x, y) in self.flower_locations and self.agent_energies[self.agent_selection] > 0:
            
            self.flower_locations.remove((x, y))
            self.num_flowers -= 1
            self.rewards[self.agent_selection] += 20
            # it can only go to 3 flowers before going back to the hive
            # one pollen gives one honey (1:1)
            self.agent_energies[self.agent_selection] -= 1
            
            if self.season != SEASONS[2] or self.season != SEASONS[3]:
                self.agent_pollination[self.agent_selection] += 1
            self.check_energy(self.agent_selection)
            
    # kills the bee if it's a pesticide flower
    def check_pesticide(self, pos):
        (x, y) = pos

        if (x, y) in self.pesticides_locations:
            self.pesticides_locations.remove((x, y))
            self.rewards[self.agent_selection] -= 1
            self.dones[self.agent_selection] = True
            self.agent_locations[self.agent_selection] = (-1, -1)
            self.num_pesticides -= 1
            return True
        return False

    def check_in_hive(self, pos):
        (x, y) = pos
        if (x, y) == (0, 0):
            pollen = 3 - self.agent_energies[self.agent_selection]
            self.num_honey += pollen
            if (self.agent_energies[self.agent_selection] <= 0):
                self.agent_energies[self.agent_selection] = 3
            return True
        return False

    # If the energy is 0, the bee goes back to the hive
    # means the bee has picked up pollen from 3 flowers
    def check_energy(self, agent):
        return self.agent_energies[agent]

    def check_pollination(self, agent):
        return self.agent_pollination[agent]

    # returns the number of agents that haven't dies yet
    def num_alive_agents(self):
        count = 0
        for agent in self.agents:
            if not self.dones[agent]:
                count += 1
        return count

    def num_agents_on(self, x, y):
        count = 0
        for agent in self.agents:
            if self.agent_locations[agent] == (x, y):
                count += 1
        return count

    def print_info(self):
        print("Season: " + str(self.season))
        print("Locations: ", self.agent_locations)
        print("Num Alive Agents: " + str(self.num_alive_agents()))
        print("Energies: ", self.agent_energies)
        print("Steps: " + str(self.steps))
        print("Level of Honey: " + str(self.num_honey))
        print()
        #sleep(0.25)

    def simplified_features(self):

        agent_pos = []
        for agent in self.agents:
            row, col = self.agent_locations[agent]
            agent_pos.append((col, row))

        flower_pos = []
        for flower_id in range(len(self.flower_locations)):
            row, col = self.flower_locations[flower_id]
            flower_pos.append((col, row))

        pesticide_pos = []
        for pesticide_id in range(len(self.pesticides_locations)):
            row, col = self.pesticides_locations[pesticide_id]
            pesticide_pos.append((col, row))

        features = [agent_pos, flower_pos, pesticide_pos]

        return features