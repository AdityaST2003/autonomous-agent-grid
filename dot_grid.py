import pygame
import random
import numpy as np

# Grid dimensions
GRID_SIZE = 10
CELL_SIZE = 50
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 50, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)

# Cell types
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3

# Create environment
def create_grid():
    grid = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for _ in range(15):
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if (x, y) not in [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]:
            grid[y][x] = OBSTACLE
    grid[0][0] = START
    grid[GRID_SIZE - 1][GRID_SIZE - 1] = GOAL
    return grid

def draw_grid(screen, grid, agent_pos):
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[y][x] == OBSTACLE:
                color = BLACK
            elif grid[y][x] == START:
                color = BLUE
            elif grid[y][x] == GOAL:
                color = GREEN
            else:
                color = WHITE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)
    ax, ay = agent_pos
    agent_rect = pygame.Rect(ax * CELL_SIZE + 10, ay * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, RED, agent_rect)

class QLearningAgent:
    def __init__(self, grid_size, num_actions):
        self.q_table = np.zeros((grid_size, grid_size, num_actions))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        x, y = state
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[y][x])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        current = self.q_table[y][x][action]
        max_future = np.max(self.q_table[ny][nx])
        new_value = (1 - self.alpha) * current + self.alpha * (reward + self.gamma * max_future)
        self.q_table[y][x][action] = new_value

def move_agent(agent_pos, direction, grid):
    x, y = agent_pos
    if direction == "UP" and y > 0 and grid[y - 1][x] != OBSTACLE:
        y -= 1
    elif direction == "DOWN" and y < GRID_SIZE - 1 and grid[y + 1][x] != OBSTACLE:
        y += 1
    elif direction == "LEFT" and x > 0 and grid[y][x - 1] != OBSTACLE:
        x -= 1
    elif direction == "RIGHT" and x < GRID_SIZE - 1 and grid[y][x + 1] != OBSTACLE:
        x += 1
    return [x, y]

def replay(agent, grid):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Replay")

    agent_pos = [0, 0]
    path = [tuple(agent_pos)]
    clock = pygame.time.Clock()
    running = True
    steps = 0

    while running:
        screen.fill(WHITE)
        draw_grid(screen, grid, agent_pos)

        for px, py in path:
            pygame.draw.circle(screen, (200, 100, 100),
                               (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2),
                               5)

        pygame.display.flip()
        clock.tick(5)

        state = tuple(agent_pos)
        action = np.argmax(agent.q_table[state[1]][state[0]])
        new_pos = move_agent(agent_pos[:], ["UP", "DOWN", "LEFT", "RIGHT"][action], grid)
        agent_pos = new_pos
        path.append(tuple(agent_pos))
        steps += 1

        cell = grid[agent_pos[1]][agent_pos[0]]
        if cell in [GOAL, OBSTACLE] or steps > 100:
            pygame.time.wait(1000)
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D RL Grid Agent")

    episodes = 500
    agent = QLearningAgent(GRID_SIZE, 4)

    for episode in range(episodes):
        grid = create_grid()
        agent_pos = [0, 0]
        total_reward = 0

        for step in range(100):
            screen.fill(WHITE)
            draw_grid(screen, grid, agent_pos)
            pygame.display.flip()

            state = tuple(agent_pos)
            action = agent.choose_action(state)
            next_pos = move_agent(agent_pos[:], ["UP", "DOWN", "LEFT", "RIGHT"][action], grid)
            next_state = tuple(next_pos)

            cell = grid[next_pos[1]][next_pos[0]]
            if cell == GOAL:
                reward = 100
                done = True
            elif cell == OBSTACLE:
                reward = -100
                done = True
            else:
                reward = -1
                done = False

            agent.update(state, action, reward, next_state)
            agent_pos = next_pos
            total_reward += reward

            if done:
                break

            pygame.time.wait(10)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    pygame.quit()

    # Replay after training
    replay_grid = create_grid()
    replay(agent, replay_grid)

if __name__ == "__main__":
    main()
