from simulation_framework import *
import pygame as pg
from sarsamouse import SarsaMouse
import pickle

# Used to determine how many frames are skipped.
# Helps when you want the gamelogic to move faster than
# Your system can draw it.
SKIP_FRAMES = 000

# Number of frames to draw per second.
FRAMES_PER_SECOND = 7

pg.init()

number_of_episodes = 10000
autosave_interval = 10  # Saves agent after this many episodes
agent = SarsaMouse()
game_window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
game_window.fill(BACKGROUND_COLOR)
pg.display.set_caption('Simulation')


def GameLoop(game_manager, mouse):
    paused = False

    frameCount = 0
    clock = pg.time.Clock()
    run_game_loop = True

    player_move = mouse.getAction(game_manager.main_agent.sense)
    last_player_score = 0.

    while run_game_loop:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run_game_loop = False
                if event.key == pg.K_p:
                    paused = not paused
                if event.key == pg.K_b:
                    print("{} / {}".format(len([(k, v) for k, v in mouse.QE.items() if v[0] != 0.0]), len(mouse.QE)))
            # Check to see if the user has requested that the game end.
            if event.type == pg.QUIT:
                agent.save()
                pg.quit()
                pg.display.quit()
                exit(0)

        if not paused:
            game_manager.logicTick(player_move)
            player_move = mouse.getAction(game_manager.main_agent.sense)
            reward = game_manager.main_agent.deltaEnergy + game_manager.main_agent.deltaDamage
            mouse.update(reward)
            game_manager.main_agent.deltaDamage = game_manager.main_agent.deltaEnergy = 0.
            if not game_manager.main_agent.alive:
                mouse.getAction(game_manager.main_agent.sense)
                mouse.update(-(1000. - last_player_score))
                run_game_loop = False
            if game_manager.main_agent.score >= 1000.:
                mouse.getAction(game_manager.main_agent.sense)
                mouse.update(1000)
                run_game_loop = False
            if SKIP_FRAMES == 0 or frameCount % SKIP_FRAMES == 0:
                game_window.fill(BACKGROUND_COLOR)
                game_manager.draw(game_window, mouse)
                pg.display.flip()

            frameCount += 1

            delta_time = clock.tick(FRAMES_PER_SECOND)

highScore = 0
for k in range(number_of_episodes):
    # initialize the game manager.
    gm = GameManager(GAME_GRID_WIDTH, GAME_GRID_HEIGHT, k)
    GameLoop(gm, agent)
    if gm.main_agent.score > highScore:
        highScore = gm.main_agent.score
    if k % autosave_interval == 0:
        agent.save()
        print(f"Agent saved! High score: {highScore}")
    agent.decayEpsilon(k + 1)

agent.save()

pg.display.quit()
pg.quit()