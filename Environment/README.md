# Bee Hive Environment
A custom pettingzoo environment that simulates a very simple bee hive, based on the [ants](https://github.com/chorsch/ants) and [traffic junction](https://github.com/koulanurag/ma-gym/tree/master/ma_gym/envs/traffic_junction) pettingzoo environments.

The environment consists of a basic grid world where each cell might:
- Be empty
- Have a flower (and pollen is collected if this cell is entered)
- Have a flower with pesticide (which acts as a trap and bees are killed)
- Have one or more bees

The cell (0, 0) is a special cell since it's the hive.

**Actions**

Each time step, a bee can move in the following directions:
- Up
- Down
- Left
- Right

These steps were already developed to some extent in the ants environment.

**The Visual Interface**

The visual interface is created from the board matrix using ma-gym, more specifically code from the traffic junction environment.

**Observations**

The simulation ends when all bees have died, when they run out of honey in the winter season or when the year ends.
The more detailed description of how this environment works is described on the project's final report.
