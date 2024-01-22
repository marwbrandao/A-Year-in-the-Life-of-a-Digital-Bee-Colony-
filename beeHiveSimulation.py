import argparse


from scipy.spatial.distance import cityblock
import numpy as np
from gym import Env
from typing import List, Tuple
from typing import Sequence
import random
import math
from Agents.agent import Agent
from Environment.beeHiveEnv import BeeHiveEnv
from utils import compare_results, plot_bar
import numpy.ma as ma 
N_ACTIONS = 4
LEFT, RIGHT, UP, DOWN = range(N_ACTIONS)

MAX_AGENTS = 8

class RandomAgent(Agent):

    def __init__(self, agent_id, n_actions: int, n_agents):
        super(RandomAgent, self).__init__("Random Agent")
        self.agent_id  = agent_id
        self.n_actions = n_actions
        self.name = 'Random'
        self.n_agents= n_agents

    def action(self) -> int:
        if environment.check_energy(environment.agents[self.agent_id]) > 0 :
            return np.random.randint(self.n_actions)
        else: 
            direction_to_go(self.observation[self.agent_id][0][self.agent_id], (0, 0))

class GreedyAgent(Agent):
    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, agent_id, n_agents):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS

    def action(self) -> int:

        if environment.dones[str(self.agent_id)] == False:

            agents_positions = self.observation[self.agent_id][0]
            flower_positions = self.observation[self.agent_id][1]
            pest_positions = self.observation[self.agent_id][2]
            agent_position = agents_positions[self.agent_id]
            agent = environment.agents[self.agent_id]

            if environment.check_energy(agent) > 0:
                closest_flower = closest_prey(agent_position, flower_positions + pest_positions)
                prey_found = closest_flower is not None
                return direction_to_go(agent_position, closest_flower, ) if prey_found else random.randrange(
                    N_ACTIONS)
            else:
                return direction_to_go(agent_position, (0, 0))

    # ################# #
    # Auxiliary Methods #
    # ################# #

    # def direction_to_go(self, agent_position, prey_position):
    #     """
    #     Given the position of the agent and the position of a prey,
    #     returns the action to take in order to close the distance
    #     """
    #     distances = np.array(prey_position) - np.array(agent_position)
    #     abs_distances = np.absolute(distances)
    #     if abs_distances[0] > abs_distances[1]:
    #         return self._close_horizontally(distances)
    #     elif abs_distances[0] < abs_distances[1]:
    #         return self._close_vertically(distances)
    #     else:
    #         roll = random.uniform(0, 1)
    #         return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

def closest_prey(agent_position, prey_positions):
    """
    Given the positions of an agent and a sequence of positions of all prey,
    returns the positions of the closest prey
    """
    min = math.inf
    closest_prey_position = None
    n_preys = len(prey_positions)

    for p in range(n_preys):
        prey_position = prey_positions[p]
        distance = cityblock(agent_position, prey_position)
        if distance < min:
            min = distance
            closest_prey_position = prey_position
    return closest_prey_position

    # # ############### #
    # # Private Methods #
    # # ############### #

    # def _close_horizontally(self, distances):
    #     if distances[0] > 0:
    #         return RIGHT
    #     elif distances[0] < 0:
    #         return LEFT
    #     else:
    #         return random.randrange(LEFT, RIGHT)

    # def _close_vertically(self, distances):
    #     if distances[1] > 0:
    #         return DOWN
    #     elif distances[1] < 0:
    #         return UP
    #     else:
    #         return random.randrange(UP, DOWN)

class ConventionAgent(Agent):
    """
       The agent gets a predefined area to search on, depending on the number of alive agents.
       Then searches on that area, like a greedy agent.
       """

    def __init__(self, agent_id, n_agents, convention: List):
        super(ConventionAgent, self).__init__(f"Convention Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.conventions = convention

    def action(self) -> int:

        if environment.dones[str(self.agent_id)] == False:

            # agent_order = self.conventions[0]
            action_order = self.conventions
            agents_positions = self.observation[self.agent_id][0]
            flower_positions = self.observation[self.agent_id][1]

            pest_positions = self.observation[self.agent_id][2]
            agent_position = agents_positions[self.agent_id]
            agent = environment.agents[self.agent_id]
            
            dir = self.agent_id % 4

            if environment.check_energy(agent) > 0:
                closest_flower = closest_prey_conventions(agent_position, flower_positions + pest_positions,
                                                   action_order[dir], self.conventions)
                if closest_flower != math.inf:
                    return direction_to_go(agent_position, closest_flower
                                                )
                else: return random.randrange(N_ACTIONS)
            else:
                return direction_to_go(agent_position, (0, 0))

# ################# #
# Auxiliary Methods #
# ################# #

def direction_to_go(agent_position, prey_position):
    """
    Given the position of the agent and the position of a prey,
    returns the action to take in order to close the distance
    """
    distances = np.array(prey_position) - np.array(agent_position)
    abs_distances = np.absolute(distances)
    
    if abs_distances[0] > abs_distances[1]:
        return close_horizontally(distances)
    elif abs_distances[0] < abs_distances[1]:
        return close_vertically(distances)
    else:
        roll = random.uniform(0, 1)
        return close_horizontally(distances) if roll > 0.5 else close_vertically(distances)

def closest_prey_conventions(agent_position, prey_positions, agent_dest: int, conventions):
    """
    Given the positions of an agent and a sequence of positions of all prey,
    returns the positions of the closest prey
    """

    _min = math.inf
    closest_prey_position = math.inf
    n_preys = len(prey_positions)
    grid = environment._grid_shape

    grid_spacing0 = grid[0] / len(conventions) * agent_dest
    grid_spacing1 = grid[1] / len(conventions) * agent_dest
    
    for p in range(n_preys):
        prey_position = prey_positions[p]
        
        if prey_position[0] >= grid_spacing0 or prey_position[1] >= grid_spacing1:  # only searches for preys in the designated area
            distance = cityblock(agent_position, prey_position)
            if distance < _min:
                _min = distance
                closest_prey_position = prey_position
        else:
            continue

    return closest_prey_position

# ############### #
# Private Methods #
# ############### #

def close_horizontally(distances):
    if distances[0] > 0:
        return RIGHT
    elif distances[0] < 0:
        return LEFT
    else:
        return random.randrange(LEFT, RIGHT)

def close_vertically(distances):
    if distances[1] > 0:
        return DOWN
    elif distances[1] < 0:
        return UP
    else:
        return random.randrange(UP, DOWN)

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:
    results = np.zeros(n_episodes)

    for episode in range(n_episodes):
        print(f"Episode {episode} out of {n_episodes} completed.")

        steps = 0
        #terminals = [False for _ in range(len(agents))]
        terminal = False
        observations = environment.reset()
        while not terminal:
            steps += 1

            for agent in agents:
                agent.see(observations)

            actions = [agent.action() for agent in agents]
            next_observations = []
            
            for action in actions:
                next_observation , terminal = environment.step(action)
                next_observations.append(next_observation)
                #print(terminal)
                #terminals[i] = terminal
                #i+=1
            environment.render()
            #print(step_info)
            #next_observations=np.array(step_info)[:,0]
            #terminals=np.array(step_info)[:, 1]
            observations = next_observations
        print("final num steps " + str(steps))
        results[episode] = steps
        environment.close()

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = BeeHiveEnv(n_agents = MAX_AGENTS, grid_shape = (30, 30))

    # 2 - Setup agents
    convenctions = [0, 1, 2, 3]
    teams = {

        "Random Team": [
           RandomAgent(agent_id = i, n_actions= environment.n_actions, n_agents=MAX_AGENTS) for i in range(MAX_AGENTS)],

        "Greedy Team": [
        GreedyAgent(agent_id=i, n_agents=MAX_AGENTS) for i in range(MAX_AGENTS)],

        "Greedy Team \nw/ Social Convention": [
            ConventionAgent(agent_id=i, n_agents=MAX_AGENTS, convention=convenctions) for i in range(MAX_AGENTS)
        ]
    }

    results = {}
    for team, agents in teams.items():
        print("Running team " + str(team))
        result = run_multi_agent(environment, agents, opt.episodes)
        results[team] = result
    
    
    survival_rate=[]
    for team in teams:
        arr = ma.masked_less(results[team], 365)
        survival_rate.append(arr.sum()/365)
    

    compare_results(results, title="Hive survival", colors=["orange", "blue", "green"])
    plot_bar(list(results.keys()),survival_rate, "Survival rate", "", colors=["orange", "blue", "green"])
