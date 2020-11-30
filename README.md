# AI_Assignment_2
Assignment 2 source code for AI with Reinforcement Learning Course


## READ THIS FIRST
I'm still fixing some aspects of the code. It'll be finished ASAP, but please let me know if there is any issues. Please open an issue in this GitHub to get the issue fixed quickly. 

## Methods
You are to use one of the tabular methods that have been taught in class (But these are recomended):
1.	Temporal Difference
2.	SARSA (State-Action-Reward-State-Action)

## Simulation Notes
There are a few different factors to concider in this assignment

### Enviornmental Factors
Environmental Factors that must be considered:
1. Terrain Difficulty:
   * The terrain now has hills and valleys. As in real life, moving up a hill uses more energy than moving down a hill.  
2. Adversary:
    * There is now a predator who will randomly wander around. It can see you if you are within three tiles of it. If you are, it will move towards you. If you end your turn next to it, it will begin to sap HP (Health Points) away from you. Your HP will gradually recover over time at the expense of your energy.
3. Food Growth:
   * Food no longer spawns at its full energy capacity. Instead, it must grow over time until fully grown.
   * When a plant is eaten, another plant will spawn somewhere else on the grid.
   * Plant ‘smell’ is no longer infinite and is now capped. 

### New Features:
1. Diagonal movement is now possible. 
   * Moving diagonally incurs a base energy use cost of sqrt (2) instead of 1, as it is a longer distance than a cardinal movement on a square grid. 
   * This would not be the case on a hexagon grid, though the matricies would be too complicated, so we're sticking to the square grid, which is unfortunate because [Hexagons are the Bestagons.](https://www.youtube.com/watch?v=thOifuHs6eY)
2. Let there be sight! Your creature now has a limited range of sight, in addition to its original ability to smell its direct surroundings.
   * See the **Agent Abilities** section of this README for more information.
4. Game states are now slightly easier to restore. Just take a snapshot by creating a new GameState object with the current GameManager object as a parameter (usually just gm), and you'll have a copy of the current game state. 
   * Restore it by calling the GameState.restore function with the current GameManager object as a parameter (again, usually just gm). 
   * The game won't play out the same each time, but let me know if you'd like a deterministic mode.
     
### Agent Abilities
There are two types of agents; a herbivore class and a carnivore class. Both creatures will, by default, follow their nose to try to find food in the same way that smart_mouse did in assignment 1. They are not as smart as smart_mouse, and this intelligence is measured by the DEFAULT_INTELLIGENCE and DEFAULT_EVIL_INTELLIGENCE stats. Below is a description of each creature type.

**Herbivore** ![Herbivore](src/art_assets/agent_faces/agent_faces_neutral.png)

These creatures include the player, and are peaceful. All they do is wander around and try to find food.

To differentiate your player from the generic creatures, you have a cool pair of sunglasses.
![Sunglasses](src/art_assets/agent_faces/agent_faces_main.png)

**Carnivore (Evil)** ![Carnivore](src/art_assets/agent_faces/agent_faces_evil.png)

While not actually evil, they're effectively evil towards the player, as they will sniff you out and hunt you down. Or any of the other herbivores. Which ever it finds first.

**Plants** ![Plant](src/art_assets/plant_growth/plant5.png)

More of an object than an agent. Will grow over time, and be both more fragrant and more energizing to eat. Can be eaten in pieces or all at once using the EAT_PLANT_INSTANT boolean.

**Senses**

Both Herbivores and carnivores have the ability to see the ground below them in a 5x5 grid, along with three other 5x5 grids that tells them wether a nearby tile has something on it. It can tell the difference between food (Plants), other creatures, and if a creature is dangerous.

Alongside sight, they also have a 3x3 smell matrix that gives them information about nearby food or creatures. They do not have the ability to tell a good creature from a bad creature just by their sense of smell unless they're a carnivore.

**Special Notes**

1. A herbivore will not attack anything other than plants.
2. A carnivore will always attack herbivores.
3. Herbivores and carnivores both follow their noses, but herbivores follow their food sense, while carnivores follow their creature sense.

## Your Tasks
1)	Each turn, give your agent a movement number between 0 and 8, corresponding to the following table (Note that a value of 4 will make the agent stand still for the turn):
![Directions](readme_imgs/directions.png)

    * All you have to do is find the player_move line in simulation_runner.py, and change it from None to the move you want to make that turn.

 2) You will be given a set of three scenarios that can be created by modifying global variables. You're task will be to accomplish the given task in that scenario.

## Scenarios
 
These scenarios will have more details soon to help you out and give you hints as to how to solve them.
Be sure to update your code frequently.

### Scenario 1: Terrain Navigation
**Setup**

NUM_AGENTS = 1

NUM_EVIL = 0

**Details**
This is very similar to your tasks for Assignment 1. Your job is to give the player character the ability to navigate the new terrain. 


### Scenario 2: Aggression Avoidance
**Setup**

NUM_AGENTS = 1

NUM_EVIL = 1

**Details**
In addition to Scenario 1's task, the player must now deal with an aggressive agent.

### Scenario 3: Competition
**Setup**

NUM_AGENTS = 5

NUM_EVIL = 1

**Details**
In addition to Scenario 2's task, the player now has to compete for food with other herbivores. The biggest challenge here is that the player cannot differentiate the scent of herbivores from carnivores, but they do have the ability to tell the difference in their sight.

### Questions
Please answer these questions to the best of your abilities. They won't be graded on correctness, but more regarding the amount thought you put into them. It's not difficult to get full points on each question.

**(IN PROGRESS: CHECK BACK SOON)**

### Scenario 1:

1) Do you expect your player to perform better with a 5x5 sight matrix for the terrain vs a 3x3? How far ahead does your model 'plan' its movements?

2) Do you expect your player to tend towards staying in the low ground, or the high ground? Would this change with different terrain patterns?

3) Do you expect a model trained with a specific terrain map to perform the same as on a different map?


### Scenario 2:

1) Considering only the scent matrices for food and other agents, how would you rewrite smart mouse to perform better in this scenario?

2) In Scenarios 1 and 2, is it necessary to use the agent and danger **sight** matrices? Why or why not?


### Scenario 3: 
   1) Do you expect your player to still take terrain navigation into account, or do you think other factors will take priority?

   2) What effect does having multiple neutral agents (That also want the same resource that you do) also show up in the creature smell matrix have on the simulation? Why does this make the danger sight matrix useful? 


## Extra Credit

Extra credit is awarded for creating, implementing, and solving a unique scenario. 

For example, you could create a poison plant that the creatures have to avoid, or create a custom height map for navigation. If you need a specific function, but don't have the technical ability, let me know and I'll see if I can add it.
