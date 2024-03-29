import argparse
import random
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cityblock
from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from lab1_solutions.exercise_1_single_random_agent import RandomAgent
from lab1_solutions.exercise_2_single_random_vs_greedy import GreedyAgent
from lab1_solutions.exercise_3_multi_agent import run_multi_agent

from exercise_2_social_conventions import ConventionAgent

ACTIONS = 4
GO_NORTH, GO_SOUTH, GO_WEST, GO_EAST = range(ACTIONS)

MOVEMENTS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(MOVEMENTS)


class RoleAgent(Agent):
    
    def __init__(self, agent_id: int, n_agents: int, roles: List, role_assign_period: int = 1):
        super(RoleAgent, self).__init__(f"Role-based Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.roles = roles
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0

    def potential_function(self, agent_pos: Tuple, prey_pos: Tuple, role: int):
        """
        Calculates the potential function used for role assignment.
        The potential function consists of the negative Manhattan distance between the
        `agent_pos` and the target position of the given `role` (which corresponds
        to a position that is adjacent to the position of the prey).

        :param agent_pos: agent position
        :param prey_pos: prey position
        :param role: role

        :return: (float) potential value
        """
        prey_adj_locs = self.get_prey_adj_locs(prey_pos)
        role_target_pos = prey_adj_locs[role]
        return -cityblock(agent_pos, role_target_pos)
        # TODO: Calculate the potential function given by the negative Manhattan distance
        # between the agent position and the target position given the role.
        # distance=
        # for i in range(len(prey_adj_locs)):
        #     distance += cityblock(agent_pos, prey_adj_locs[i])
        
        #raise ValueError("Not implemented.")

    def role_assignment(self):
        """
        Given the observation vector containing the positions of all predators
        and the prey(s), compute the role-assignment for each of the agents.

        :return: a list with the role assignment for each of the agents
        """
        prey_pos = self.observation[self.n_agents * 2:]
        agent_positions = self.observation[:self.n_agents * 2]
        
        # TODO: first calculate the potential function for each agent-role pair.
        # Then, iteratively assign, given the fixed role order, a role to each
        # of the agents. Make sure that, in case more than one agent has the same
        # potential value for a given role, the agent selection is done is a
        # consistent (deterministic) manner.
        #raise ValueError("Not implemented.")
        assigned_roles=[0 for i in range(self.n_agents)]
        potentials=[]
        state=[0 for i in range(self.n_agents)]
        
        #for i in range(self.n_agents):
        print(self.roles)
        for role in self.roles:
            print(role)
            successful = False
            #agent=agent
            potentials = [self.potential_function((agent_positions[i*2], agent_positions[i*2+1]), prey_pos, role) for i in range(self.n_agents)]
            while not successful:
                print("hi")
                agent=np.argmax(potentials)
                if state[agent]==0:
                    assigned_roles[role]=agent
                    state[agent]=1
                    successful=True
                    print("olá")
                else: 
                    potentials[agent]=-9999
            print(successful)
            
            #potential=-10000
            #agent=-1
            # for i in range(self.n_agents):
            #     if potential < self.potential_function((agent_positions[i*2], agent_positions[i*2+1]), prey_pos, role):
            #         potential = self.potential_function((agent_positions[i*2], agent_positions[i*2+1]), prey_pos, role)
            #         agent=i
            # assigned_roles.insert(agent, role)   
        print(assigned_roles) 
        return assigned_roles

    def action(self) -> int:

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            role_assignments = self.role_assignment()
            self.curr_role = role_assignments[self.agent_id]
        print(self.curr_role)
        prey_pos = self.observation[self.n_agents * 2:]
        agent_pos = (self.observation[self.agent_id * 2], self.observation[self.agent_id * 2 + 1])
        self.steps_counter += 1
        
        return self.advance_to_pos(agent_pos, prey_pos, self.curr_role)

    def get_prey_adj_locs(self, loc: Tuple) -> List[Tuple]:
        prey_x = loc[0]
        prey_y = loc[1]
        return [(prey_x, prey_y - 1), (prey_x, prey_y + 1), (prey_x - 1, prey_y), (prey_x + 1, prey_y)]

    def advance_to_pos(self, agent_pos: Tuple, prey_pos: Tuple, agent_dest: int) -> int:
        """
        Choose movement action to advance agent towards the destination around prey
        
        :param agent_pos: current agent position
        :param prey_pos: prey position
        :param agent_dest: agent destination in relation to prey (0 for NORTH, 1 for SOUTH,
                            2 for WEST, and 3 for EAST)

        :return: movement index
        """

        def _move_vertically(distances) -> int:
            if distances[1] > 0:
                return DOWN
            elif distances[1] < 0:
                return UP
            else:
                return STAY
            
        def _move_horizontally(distances) -> int:
            if distances[0] > 0:
                return RIGHT
            elif distances[0] < 0:
                return LEFT
            else:
                return STAY
            
        prey_adj_locs = self.get_prey_adj_locs(prey_pos)
        distance_dest = np.array(prey_adj_locs[agent_dest]) - np.array(agent_pos)
        abs_distances = np.absolute(distance_dest)
        if abs_distances[0] > abs_distances[1]:
            return _move_horizontally(distance_dest)
        elif abs_distances[0] < abs_distances[1]:
            return _move_vertically(distance_dest)
        else:
            roll = np.random.uniform(0, 1)
            return _move_horizontally(distance_dest) if roll > 0.5 else _move_vertically(distance_dest)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SimplifiedPredatorPrey(grid_shape=(15, 15), n_agents=4, n_preys=1, max_steps=100, required_captors=4)

    # Set seeds.
    random.seed(3)
    np.random.seed(3)
    environment.seed(3)

    # 2 - Setup the teams
    conventions = [[0, 1, 2, 3], [GO_NORTH, GO_SOUTH, GO_WEST, GO_EAST]]

    roles = [GO_NORTH, GO_SOUTH, GO_WEST, GO_EAST]
    teams = {

        "Random Team": [
                RandomAgent(environment.action_space[0].n),
                RandomAgent(environment.action_space[1].n),
                RandomAgent(environment.action_space[2].n),
                RandomAgent(environment.action_space[3].n),
        ],

        "Greedy Team": [
                GreedyAgent(agent_id=0, n_agents=4),
                GreedyAgent(agent_id=1, n_agents=4),
                GreedyAgent(agent_id=2, n_agents=4),
                GreedyAgent(agent_id=3, n_agents=4),
        ],

        "Greedy Team \nw/ Social Convention": [
                ConventionAgent(agent_id=0, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=1, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=2, n_agents=4, social_conventions=conventions),
                ConventionAgent(agent_id=3, n_agents=4, social_conventions=conventions),
        ],

        "Greedy Team \nw/ Roles": [
                RoleAgent(agent_id=0, n_agents=4, roles=roles),
                RoleAgent(agent_id=1, n_agents=4, roles=roles),
                RoleAgent(agent_id=2, n_agents=4, roles=roles),
                RoleAgent(agent_id=3, n_agents=4, roles=roles),
        ],

    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        print(f'Running {team}.')
        result = run_multi_agent(environment, agents, opt.episodes)
        results[team] = result

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Predator Prey' Environment",
        colors=["orange", "green", "blue", "gray"]
    )
