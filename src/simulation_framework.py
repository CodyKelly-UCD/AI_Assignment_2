import pygame as pg
import numpy as np
import random
import sys
from os import path
from PIL import Image, ImageFilter
from variable_config import *
#import time
from enum import Enum
from math import sqrt
from copy import deepcopy
#random.seed(99)

# Initialize pygame.
#pg.init()

# Get the current path of the python file. Used to load a font resource.
ABS_PATH = path.dirname(path.realpath(__file__))



def fast_dist(x1,y1,x2,y2):
    return np.linalg.norm(np.array((x1,y1))-np.array((x2,y2)))

def dir2offset(direction):
    difficulty_multiplier = 1
    x = 0
    y = 0
    d = direction
    if d >= 0 and d <= 8:
        if d in [0,1,2]:
            y = -1
        elif d in [3,4,5]:
            y = 0
        else:
            y = 1

        if d in [0,3,6]:
            x = -1
        elif d in [1,4,7]:
            x = 0
        else:
            x = 1

        # Diagonal movements are more costly than cardinal movements
        if x != 0 and y != 0:
            difficulty_multiplier = sqrt(2)
    else:
        print("Invalid direction, staying still")
    return x, y, difficulty_multiplier


# A class that allows for the saving and restoring of the game.
class GameState():
    def __init__(self, game_manager):
        self.game_manager = game_manager.deepcopy()

    def restore(self, game_manager):
        game_manager = self.game_manager

# class SensoryMatrix:
class GameObject:
    """ TODO: ADD DOCSTRING """
    def __init__(self,x,y,raw_img_path=None,stage=None):
        if raw_img_path == None:
            raw_img_path = path.join(ABS_PATH, "art_assets","ERROR")
        self.type = None
        self.difficulty = DEFAULT_TERRAIN_DIFFICULTY
        self.x = x
        self.y = y
        self.stage = stage
        self.alive = True
        self.calc_img_path(raw_img_path)
        self.loadImg(self.img_path)
        self.energy = 0    
        self.max_energy = 100

    def consume(self,energy):
        self.energy += energy
        if self.energy > self.max_energy:
            self.energy = self.max_energy

    def deplete(self,energy):
        self.energy -= energy
        if self.energy <= 0:
            self.die()
    
    def die(self):
        self.alive = False
    
    def loadImg(self, img_path):
        self.img = pg.image.load(img_path)
        self.img = pg.transform.scale(self.img,(SQUARE_SIZE,SQUARE_SIZE))
        self.img_rect = self.img.get_rect()

    def calc_img_path(self, raw_img_path):
        if self.stage is not None:
            img_path = f"{raw_img_path}{self.stage}.png"
        else:
            img_path = f"{raw_img_path}.png"
        if path.exists(img_path):
            self.img_path = img_path
        else:
            print(f"ERROR: FILE NOT FOUND ({img_path})")
            sys.exit(101)

    def move_instant(self,x,y):
        """ Move to a location without using energy """
        self.x = x
        self.y = y

    def move_probabalistic(self, movement_matrix):
        """ Input a 3x3 matrix, pick a direction based on probabilities """

        movement_list = list(range(0,9))
        movement = random.choices(movement_list,weights=movement_matrix.flatten().tolist())
        return movement[0]

    def draw(self,x,y,surface):
        surface.blit(self.img, self.img_rect.move(x,y))


class Plant(GameObject):
    def __init__(self,x=None, y=None):
        self.stage = 1
        self.raw_img_path = path.join(ABS_PATH, "art_assets","plant_growth","plant")
        super().__init__(x,y,self.raw_img_path,stage=self.stage)
        # Probability of growth per round
        self.growth_rate = 0.9
        self.num_stages = 5
        self.max_energy = 100
        self.energy = 10
        self.energy_steps = int(self.max_energy / self.num_stages)


    def tick(self):
        if random.random() < self.growth_rate:
            self.grow()

        new_stage = self.energy2stage()
        if new_stage != self.stage:
            self.stage = new_stage
            self.calc_img_path(self.raw_img_path)
            self.loadImg(self.img_path)


    def grow(self):
        self.energy += 1
        if self.energy > self.max_energy:
            self.energy = self.max_energy

    # Calculate stage based on energy level:
    def energy2stage(self):
        for i in range(self.num_stages+1):
            if self.energy <= i * self.energy_steps:
                return i
        return i


class Agent(GameObject):
    def __init__(self,x=None,y=None,raw_img_path=None):
        if raw_img_path is None:
            self.raw_img_path = path.join(ABS_PATH, "art_assets","agent_faces","agent_faces_neutral")
        super().__init__(x,y,self.raw_img_path)
        self.sense = AgentSense()
        self.movement_choice = 4
        self.max_energy = MAX_ENERGY
        self.energy = self.max_energy
        self.health = MAX_HEALTH
        self.score = 0
        self.alive = True
        self.type = 'neutral'
        self.id = random.randint(0,10000000)
        self.sense.id = self.id
        self.good_choice_chance = DEFAULT_INTELLIGENCE
        self.score = 0
        self.deltaEnergy = 0
        self.deltaDamage = 0

    def consume(self,energy):
        self.energy += energy
        self.deltaEnergy += energy
        if self.energy > self.max_energy:
            self.energy = self.max_energy
        # EC Idea: What about other ways to calculate score?

        health_score = self.health/MAX_HEALTH
        if health_score < 0.001:
            health_score = 0.001
        energy_score = self.energy/MAX_ENERGY
        if energy_score < 0.001:
            energy_score = 0.001

        self.score += energy * health_score * energy_score

    def tick(self):
        if self.energy <= 0 or self.health <= 0:
            self.die()

        if self.type == 'main' and self.alive:
            self.heal()

    def choose_movement(self):
        move = random.randint(0,8)

        if random.random() <= self.good_choice_chance:
            smell_list = list(self.sense.food_smell.flatten())
            move = smell_list.index(max(smell_list))
            if sum(smell_list) < 100:
                move = random.randint(0,8)

        return move

    def move(self,x,y,difficulty):
        self.x = x
        self.y = y
        self.energy -= difficulty
        self.deltaEnergy -= difficulty

    def heal(self):
        if self.health < MAX_HEALTH:
            self.health += 1
            self.deplete(1)
            self.calc_color()

    def die(self):
        self.raw_img_path = path.join(ABS_PATH, "art_assets","agent_faces","agent_faces_dead")
        self.calc_img_path(self.raw_img_path)
        self.loadImg(self.img_path)
        blue = 0
        if self.type == 'evil':
            blue = 255
        self.img.fill(pg.Color(255,0,blue,1),special_flags=pg.BLEND_MIN)
        self.alive = 0

    def setType(self,new_type):
        if new_type == "main":
            self.type = new_type
            self.raw_img_path = path.join(ABS_PATH, "art_assets","agent_faces","agent_faces_main")
            self.calc_img_path(self.raw_img_path)
            self.loadImg(self.img_path)
            
    def calc_color(self):
        self.loadImg(self.img_path)
        red_color =  int(255-(255 * (self.health/MAX_HEALTH)))
        if red_color < 0:
            red_color = 0
        self.img.fill(pg.Color(255,255-red_color,255-red_color,1),special_flags=pg.BLEND_MIN)


    def take_damage(self, damage):
        self.health -= damage
        self.deltaDamage = -damage * 2
        if self.health >= 0:
            self.calc_color()
        else:
            self.die()
    
    def draw(self,x,y,surface):
        surface.blit(self.img, self.img_rect.move(x,y))
        if self.type == 'main':
            self.sense.draw(surface)



# Draw the grid without anything else.
def drawGenericGrid(self,surface,rect,num_x,num_y):

    total_x = rect.width
    total_y = rect.height
    grid_pos_x = rect.x
    grid_pos_y = rect.y
    
    line_width = 1
    square_size = int(rect.width/num_x)
    line_color = pg.Color("#000000")

    for i in range(num_y + 1):
        pg.draw.rect(
                    surface,
                    line_color,
                    pg.Rect(
                        grid_pos_x,
                        grid_pos_y, 
                        1, 
                        total_y)
                )
        grid_pos_x += square_size
        if num_x == 3 and i == 2:
            grid_pos_x += 1

    grid_pos_x = rect.x

    for i in range(num_x + 1):
        pg.draw.rect(
                    surface,
                    line_color,
                    pg.Rect(
                        grid_pos_x,
                        grid_pos_y, 
                        total_x, 
                        1)
                )
        grid_pos_y += square_size
        if num_y == 3 and i == 2:
            grid_pos_y += 1


    # grid_pos_y = self.padding + self.grid_padding
    
    # for i in range(self.width + 1):
    #     pg.draw.rect(
    #                 surface,
    #                 self.line_color,
    #                 pg.Rect(
    #                     self.padding + self.grid_padding,
    #                     grid_pos_y, 
    #                     total_x,
    #                     self.padding
    #                     )
    #             )
    #     grid_pos_y += self.square_size + self.padding


class AgentSense:
    def __init__(self):
        self.sm_font = pg.font.Font(path.join(ABS_PATH,"Retron2000.ttf"), 11)

        self.sight_dist_from_agent = 2
        self.smell_dist_from_agent = 1

        self.sight_range = self.sight_dist_from_agent * 2 + 1
        self.smell_range = self.smell_dist_from_agent * 2 + 1

        self.reset_sight()
        self.reset_smell()
        
        self.sight_rects = []
        self.smell_rects = []


        self.type = "neutral"
        
        for i in range(4):
            sight_rect = pg.Rect(
                            10 + 60 * i,
                            WINDOW_HEIGHT - 60,
                            50,
                            50
                            )
            self.sight_rects.append(sight_rect)


        for i in range(2):
            smell_rect = pg.Rect(
                            10 + 60 * 4 + 60 * i,
                            WINDOW_HEIGHT - 60,
                            50,
                            50
                            )
            self.smell_rects.append(smell_rect)


    def reset_sight(self):
        self.elevation_sight = np.zeros((self.sight_range,self.sight_range))
        self.food_sight = np.zeros((self.sight_range,self.sight_range))
        self.creature_sight = np.zeros((self.sight_range,self.sight_range))
        self.danger_sight = np.zeros((self.sight_range,self.sight_range))

    def apply_sight_to_array(self):
        self.sight_senses = []

        self.sight_senses.append(self.elevation_sight)
        self.sight_senses.append(self.food_sight)
        self.sight_senses.append(self.creature_sight)
        self.sight_senses.append(self.danger_sight)

    def apply_smell_to_array(self):
        self.smell_senses = []

        self.smell_senses.append(self.food_smell)
        self.smell_senses.append(self.creature_smell)

    def reset_smell(self):
        self.food_smell = np.zeros((self.smell_range,self.smell_range))
        self.creature_smell = np.zeros((self.smell_range,self.smell_range))

    def draw(self, surface):

        surface.blit(self.sm_font.render(f"Terrain", 0, (255, 0, 0)), (10, WINDOW_HEIGHT - 80))
        surface.blit(self.sm_font.render(f"Food", 0, (255, 0, 0)), (80, WINDOW_HEIGHT - 80))
        surface.blit(self.sm_font.render(f"Agent", 0, (255, 0, 0)), (135, WINDOW_HEIGHT - 80))
        surface.blit(self.sm_font.render(f"Danger", 0, (255, 0, 0)), (190, WINDOW_HEIGHT - 80))

        surface.blit(self.sm_font.render(f"Food", 0, (255, 0, 0)), (260, WINDOW_HEIGHT - 80))
        surface.blit(self.sm_font.render(f"Agent", 0, (255, 0, 0)), (320, WINDOW_HEIGHT - 80))


        for i in range(4):
            img = Image.fromarray(self.sight_senses[i]).convert('RGB')
            sense_img = pg.image.fromstring(img.tobytes("raw","RGB"), img.size, img.mode)                
            sense_img = pg.transform.scale(sense_img,(50,50))
            surface.blit(sense_img, self.sight_rects[i])
            drawGenericGrid(self,surface,self.sight_rects[i],5,5)

        for i in range(2):
            img = Image.fromarray(self.smell_senses[i]).convert('RGB')
            sense_img = pg.image.fromstring(img.tobytes("raw","RGB"), img.size, img.mode)                
            sense_img = pg.transform.scale(sense_img,(50,50))
            surface.blit(sense_img, self.smell_rects[i])
            drawGenericGrid(self,surface,self.smell_rects[i],3,3)


    def flip_matrices(self):
        for i in range(4):
            self.sight_senses[i] = np.rot90(self.sight_senses[i],2) 
            #self.sight_senses[i] = np.fliplr(self.sight_senses[i])
            #self.sight_senses[i] = np.flipud(self.sight_senses[i])
        
    def update(self,x,y,grid,agents,plants):
        self.update_sight(x,y,grid,agents,plants)
        self.update_smell(x,y,grid,agents,plants)
        grid_loc_x = 0
        for x_offset in range(-self.smell_dist_from_agent, self.smell_dist_from_agent+1):
            grid_loc_y = 0
            for y_offset in range(-self.smell_dist_from_agent, self.smell_dist_from_agent+1):
                x_new = x + x_offset
                y_new = y + y_offset
                if grid.checkValidTile(x_new,y_new):
                    for agent in agents:
                        if agent.id != self.id:
                            if not (self.type == 'evil' and agent.type == "evil"):
                                self.creature_smell[grid_loc_y,grid_loc_x] += (0.5/(fast_dist(x_new,y_new,agent.x,agent.y)+1))*255

                    for plant in plants:
                        self.food_smell[grid_loc_y,grid_loc_x] += (0.5/(fast_dist(x_new,y_new,plant.x,plant.y)+1))*(plant.energy/plant.max_energy)*255

                        
                else:
                    self.creature_smell[grid_loc_y,grid_loc_x] = 0
                    self.food_smell[grid_loc_y,grid_loc_x] = 0
                grid_loc_y += 1
            grid_loc_x += 1
        self.apply_smell_to_array()

    def update_sight(self,x,y,grid,agents,plants):
        self.reset_sight()
        self.reset_smell()

        grid_loc_x = 0
        for x_offset in range(-self.sight_dist_from_agent, self.sight_dist_from_agent+1):
            grid_loc_y = 0
            for y_offset in range(-self.sight_dist_from_agent, self.sight_dist_from_agent+1):
                x_new = x + x_offset
                y_new = y + y_offset
                if grid.checkValidTile(x_new,y_new):
                    self.elevation_sight[grid_loc_y,grid_loc_x] = grid.elevation_map[x_new,y_new]
                    self.creature_sight[grid_loc_y,grid_loc_x] = 128
                    self.danger_sight[grid_loc_y,grid_loc_x] = 128
                    self.food_sight[grid_loc_y,grid_loc_x] = 128

                    for agent in agents:
                        if agent.x == x_new and agent.y == y_new:
                            self.creature_sight[grid_loc_y,grid_loc_x] = 255
                            if agent.type == "evil":
                                self.danger_sight[grid_loc_y,grid_loc_x] = 255

                    for plant in plants:
                        if plant.x == x_new and plant.y == y_new:
                            self.food_sight[grid_loc_y,grid_loc_x] = 255

                else:
                    self.elevation_sight[grid_loc_y,grid_loc_x] = 255
                    self.creature_sight[grid_loc_y,grid_loc_x] = 0
                    self.danger_sight[grid_loc_y,grid_loc_x] = 0
                    self.food_sight[grid_loc_y,grid_loc_x] = 0

                grid_loc_y += 1
            grid_loc_x += 1
        #TODO: Cleanup this math
        self.apply_sight_to_array()
        #self.flip_matrices()

    def update_smell(self,x,y,grid,agents,plants):
        self.apply_smell_to_array()
        
class EvilAgent(Agent):
    def __init__(self,x=None,y=None):
        self.raw_img_path = path.join(ABS_PATH, "art_assets","agent_faces","agent_faces_evil")
        super().__init__(x,y,self.raw_img_path)
        self.img.fill(pg.Color("#AAAAFF"),special_flags=pg.BLEND_MIN)
        self.type = 'evil'
        self.good_choice_chance = DEFAULT_EVIL_INTELLIGENCE
        self.sense.type = 'evil'
        self.max_energy = MAX_ENERGY * 2
        self.energy = self.max_energy
    def choose_movement(self):

        move = random.randint(0,8)

        if random.random() <= self.good_choice_chance:
            smell_list = list(self.sense.creature_smell.flatten())
            move = smell_list.index(max(smell_list))
            if sum(smell_list) < 100:
                move = random.randint(0,8)

        return move

class Grid:
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.padding = 1
        self.square_size = int(WINDOW_WIDTH/GAME_GRID_WIDTH*0.8)
        self.grid_padding = self.calcGridPadding()
        self.calcGridSize()

        self.occupied_grid = np.zeros((GAME_GRID_WIDTH,GAME_GRID_HEIGHT))
        
        self.default_color = pg.Color("#FFFFFF")
        self.line_color = pg.Color("#010101")
        self.calcHeightMap()

    def calcRandNearby(self,x,y,rand_range):
        rand_range = rand_range * 2
        found = False
        empty_range = self.checkEmptyInRange(x,y,rand_range)
        if empty_range == []:
            return None, None
        tuple = random.choice(empty_range)
        return tuple[0], tuple[1]

    def checkEmptyInRange(self,x,y,rand_range):
        empties = []
        for i in range(-rand_range, rand_range + 1):
            for j in range(-rand_range, rand_range + 1):
                if self.checkValidTile(x+i,y+j):
                    if self.occupied_grid[x+i][y+j] == 0:
                        empties.append([x+i,y+j])
        return empties

    # Check to make sure a given XY set is 
    def checkValidTile(self,x,y):
        if x >= 0 and y >= 0:
            if x < GAME_GRID_WIDTH and y < GAME_GRID_HEIGHT:
                return True
        return False

    def calcHeightMap(self):
        self.elevation_map = np.random.randint(0,high=250, size=(GAME_GRID_WIDTH,GAME_GRID_HEIGHT))
        img_path = path.join(ABS_PATH,"height.png")
        img = Image.fromarray(self.elevation_map, mode='L').filter(ImageFilter.GaussianBlur(1.2))
        img.save(img_path)
        self.elevation_map = np.asarray(Image.open(img_path)).copy()
        arr_max = self.elevation_map.max()
        arr_min = self.elevation_map.min()

        for x in range(GAME_GRID_WIDTH):
            for y in range(GAME_GRID_HEIGHT):
                val = np.interp(self.elevation_map[x][y],[arr_min,arr_max],[20,255])
                self.elevation_map[x,y] = val

        img = Image.fromarray(self.elevation_map).convert('L')
        img.save(img_path)
        
        elevation_map_img = pg.image.load(img_path)
        
        self.elevation_map_img = pg.transform.scale(elevation_map_img,(self.total_x,self.total_y))
        self.elevation_map_img = pg.transform.rotate(self.elevation_map_img,90)
        self.elevation_map_img = pg.transform.flip(self.elevation_map_img,0,1)

    # Get a random valid X coordinate.
    def randGridX(self):
        return random.randint(0,GAME_GRID_WIDTH-1)

    # Get a random valid Y coordinate.
    def randGridY(self):
        return random.randint(0,GAME_GRID_HEIGHT-1)

    # Get a random valid XY coordinate set.
    def randGridSpace(self):
        return self.randGridX(), self.randGridY()

    # Efficiently get a random XY pair that isn't already used. 
    def randEmptySpace(self):
        if np.count_nonzero(self.occupied_grid) < NUM_SPACES*0.5:
            found = False
            while found == False:
                x,y = self.randGridSpace()
                if self.occupied_grid[x][y] == 0:
                    found = True
            return x,y 
        else:
            empty_left = NUM_SPACES-len(np.count_nonzero(self.occupied_grid))
            choice = random.randint(0,empty_left)
            count = 0
            for i in range(self.height):
                for j in range(self.width):
                    if self.occupied_grid[i][j] == 0:
                        if count >= choice:
                            return i,j
                        count += 1
            for i in range(self.height):
                for j in range(self.width):
                    if self.occupied_grid[i][j] == 0:
                        if count >= choice:
                            return i,j
                        count += 1
            print("ERROR: No spaces available")
            exit(9)

    # Calculate the amount of padding needed for the current grid.
    def calcGridPadding(self):
        self.total_grid_x = self.width*self.padding + self.width*self.square_size
        self.grid_padding = int((WINDOW_WIDTH - self.total_grid_x)/2)
        return self.grid_padding

    # Calculate a XY location for a given tile location
    def calcTileLocation(self,tile):
        x = tile.x * self.padding + tile.x * self.square_size + self.grid_padding
        y = tile.y * self.padding + tile.y * self.square_size + self.grid_padding
        x += self.padding*2
        y += self.padding*2
        return x, y
    
    def calcXYLocation(self,x,y):
        world_x = x * self.padding + x * self.square_size + self.grid_padding
        world_y = y * self.padding + y * self.square_size + self.grid_padding
        world_x += self.padding*2
        world_y += self.padding*2
        return world_x, world_y
        

    # Get a tile by it's coordinates. If no tile matches, return None
    def getTile(self,x,y):
        if not self.checkValidTile(x,y):
            return None
        for tile in self.occupied_spaces:
            if tile.x == x and tile.y == y:
                return tile
        return None

    def calcGridSize(self):
        self.total_x = self.width * self.padding + self.width * self.square_size
        self.total_y = self.height * self.padding + self.height * self.square_size

    # Draw the grid without anything else.
    def drawGrid(self,surface):
        grid_pos_x = self.padding + self.grid_padding
        for i in range(self.height + 1):
            pg.draw.rect(
                        surface,
                        self.line_color,
                        pg.Rect(
                            grid_pos_x,
                            self.padding + self.grid_padding, 
                            self.padding, 
                            self.total_y)
                    )
            grid_pos_x += self.square_size + self.padding

        grid_pos_y = self.padding + self.grid_padding

        for i in range(self.width + 1):
            pg.draw.rect(
                        surface,
                        self.line_color,
                        pg.Rect(
                            self.padding + self.grid_padding,
                            grid_pos_y, 
                            self.total_x,
                            self.padding
                            )
                    )
            grid_pos_y += self.square_size + self.padding

    def draw(self, surface):
        x = self.padding + self.grid_padding
        y = self.padding + self.grid_padding
        
        rect = self.elevation_map_img.get_rect().move((x,y))
        surface.blit(self.elevation_map_img, rect)


        self.drawGrid(surface)

class GameManager:
    """ A class that controls the logic and graphics of the game. """
    def __init__(self,width,height, round):
        self.grid = Grid(height, width)
        self.agents = []
        self.plants = []

        self.round = round

        self.addAgent()
        self.font = pg.font.Font(path.join(ABS_PATH,"Retron2000.ttf"), 12)

        self.agents[0].setType("main")
        self.main_agent = self.agents[0]
        for i in range(NUM_EVIL):
            self.addEvilAgent()
        for i in range(NUM_AGENTS-1):
            self.addAgent()
    
        for i in range(MAX_NUM_FOOD_ON_GRID):
            self.addPlant()
        
    def draw(self,game_window, mouse):
        self.grid.draw(game_window)
        # Draw plants
        for plant in self.plants:
            world_x, world_y = self.grid.calcXYLocation(plant.x,plant.y)
            plant.draw(world_x, world_y, game_window)

        for agent in self.agents:
                world_x, world_y = self.grid.calcXYLocation(agent.x,agent.y)
                agent.draw(world_x, world_y, game_window)
        
        
        labels_y_start = 500
        game_window.blit(self.font.render(f"HEALTH: {self.main_agent.health}", 0, (255, 0, 0)), (10, labels_y_start))
        game_window.blit(self.font.render(f"ENERGY: {round(self.main_agent.energy,2)}", 0, (255, 0, 0)), (10, labels_y_start+15))
        game_window.blit(self.font.render(f"SCORE:   {round(self.main_agent.score,2)}", 0, (255, 0, 0)), (10, labels_y_start+30))
        game_window.blit(self.font.render(f"mouse epsilon:   {round(mouse.epsilon,2)}", 0, (255, 0, 0)), (10, labels_y_start+45))
        game_window.blit(self.font.render(f"mouse alpha:   {round(mouse.alpha,2)}", 0, (255, 0, 0)), (10, labels_y_start+60))
        game_window.blit(self.font.render(f"round:   {self.round}", 0, (255, 0, 0)), (10, labels_y_start+75))
        game_window.blit(self.font.render(f"mouse updates:   {mouse.updateCount}", 0, (255, 0, 0)), (10, labels_y_start + 90))
        game_window.blit(self.font.render(f"mouse episodes completed:   {mouse.episodeCount + self.round}", 0, (255, 0, 0)), (10, labels_y_start + 105))

    def plantTick(self):
        for plant in self.plants:
            plant.tick()

    def agentTick(self,agent,move=None):
        if agent.alive == 0:
            return
        agent.tick()
        if move == None:
            move = agent.choose_movement()

        offset_x, offset_y, difficulty = dir2offset(move)
        new_x = agent.x + offset_x
        new_y = agent.y + offset_y
        
        if self.grid.checkValidTile(new_x, new_y):
            curr_height = self.grid.elevation_map[agent.x][agent.y]
            new_height = self.grid.elevation_map[new_x][new_y]
            difficulty = 1
            diff_add = (int(new_height) - int(curr_height))/255

            difficulty = DEFAULT_TERRAIN_DIFFICULTY + diff_add
            #TODO Finish this

            agent.move(new_x,new_y,difficulty)

        if agent.type != 'evil':
            for plant in self.plants:
                if agent.x == plant.x and agent.y == plant.y:
                        if EAT_PLANT_INSTANT:
                            agent.consume(plant.energy)
                            self.plants.remove(plant)
                            self.addPlant()

                        else:
                            agent.consume(10)
                            if plant.energy > 10:
                                plant.deplete(10)
                            else:
                                self.plants.remove(plant)
                                self.addPlant()

        else:
            for target_agent in self.agents:
                if target_agent.type != 'evil':
                    if target_agent.alive:
                        if agent.x == target_agent.x and agent.y == target_agent.y:
                            target_agent.take_damage(10)
                            
                    else:
                        if target_agent.energy > 10:
                            agent.consume(10)
                            target_agent.deplete(10)
                        else:
                            agent.consume(target_agent.energy)
                            self.agents.remove(target_agent)

        agent.sense.update(agent.x,agent.y,self.grid,self.agents,self.plants)

    def logicTick(self,player_move=None):
        random.shuffle(self.plants)
        random.shuffle(self.agents)
        self.plantTick()
        
        for agent in self.agents:
            if agent.type != "main":
                self.agentTick(agent)
        for agent in self.agents:
            if agent.type == "main":
                self.agentTick(agent,player_move)
        
    def addPlant(self):
        x, y = self.grid.randEmptySpace()
        plant = Plant(x,y)
        self.plants.append(plant)

    def addAgent(self):
        x, y = self.grid.randEmptySpace()
        agent = Agent(x,y)
        self.agents.append(agent)

    def addEvilAgent(self):
        x, y = self.grid.randEmptySpace()
        agent = EvilAgent(x,y)
        self.agents.append(agent)

    def setOccupiedGrid(self):
        self.grid.occupied_grid = np.zeros((GAME_GRID_WIDTH,GAME_GRID_HEIGHT))
        for plant in self.plants:
            self.grid.occupied_grid[plant.x][plant.y] = 1


# All simple mouse does is pick a random direction, and moves there.
# Quite senseless, if you ask me.
def simple_mouse():
    return random.choice(range(0,4))

# Decides wether or not to use the corners of the player sensory matrix
# when selecting a movement path. In its current state, the mouse can get
# stuck in a rut when this is true when multiple pieces of food are in play. 
# You can try to improve it, if you'd like.

# Do you think your RL model will make better use of the corners, or
# do you think it will rely on the cardinal directions that it can use
# for movement?

USE_DIAGONAL_SCENT = False


# The smart mouse uses its nose to find food. It does this by checking
# which path has the greatest amount of food smells, and going in that
# direction. 

def smart_mouse(scent_matrix):

    # If there are no scents, just pick a random direction.
    if not np.any(scent_matrix):
        return simple_mouse()
    
    # Get the maximum value, or values
    indexes = [i for i, x in enumerate(scent_matrix) if x == max(scent_matrix)]

    # Make a random choice from all the best options
    move_choice = random.choice(indexes)
    return move_choice
    # return movement_array.index(max(movement_array))


