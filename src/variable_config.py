from pygame import Color
# Window width and height
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 700

# How many tiles should the grid have horizontally and vertically?
# CURRENTLY ALL GRIDS MUST BE SQUARE
GAME_GRID_WIDTH = 15
GAME_GRID_HEIGHT = GAME_GRID_WIDTH

# Total number of spaces 
NUM_SPACES = GAME_GRID_WIDTH * GAME_GRID_HEIGHT

# Not currently used.
SMELL_PROPAGATION_DISTANCE = GAME_GRID_WIDTH

# How much energy is spent moving from tile to tile
DEFAULT_TERRAIN_DIFFICULTY = 1

# Should food smell stack, or take the greatest value?
# Glitches can occur if this is turned off
SCENT_STACKING = True

# Does a round's score go down to zero if the player dies?
DEATH_PENALTY = True

# Self explanatory
BACKGROUND_COLOR = Color("#505050")
PAUSED_BACKGROUND_COLOR = Color("#303030")



# Used to determine how many frames are skipped.
# Helps when you want the gamelogic to move faster than
# Your system can draw it.
SKIP_FRAMES = 0

# The number of food pieces that will spawn each time there is no food
# on the grid.
MAX_NUM_FOOD_ON_GRID = int(NUM_SPACES / 20)

if MAX_NUM_FOOD_ON_GRID <= 1:
    MAX_NUM_FOOD_ON_GRID = 1

# How much food needs to be found before a round ends?
FOOD_PER_ROUND = MAX_NUM_FOOD_ON_GRID * 5

# What is the maximum amount of energy that a player can have?
MAX_ENERGY = GAME_GRID_WIDTH * 10

# What is the maximum amount of health that a player can have?
MAX_HEALTH = 100

# Maximum number of game states that can be saved per round
MAX_SAVED_GAME_STATES = MAX_ENERGY

SMELL_DIST = 10

SQUARE_SIZE = int(WINDOW_WIDTH/GAME_GRID_WIDTH*0.8)

NUM_AGENTS = 1

NUM_EVIL = 0

# Defines how smart the non-student agents are.
# A value between 0 and 1, where a lower value means the 
# agent will make a good choice less often
DEFAULT_INTELLIGENCE = 0.2

# Same as above, fut only applies to evil agents
DEFAULT_EVIL_INTELLIGENCE = 0.5


# Indicates whether or not it takes time to consume a plant
EAT_PLANT_INSTANT = True