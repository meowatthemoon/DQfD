import pygame
import numpy as np
import time
from random import randrange


class Wall():
    def __init__(self, rows, cols, screen_width, screen_height):
        width = screen_width // cols
        height = int(screen_height * 0.1)
        self.blocks = []
        self.blocks_exist = []
        for row in range(rows):
            block_row = []
            block_exist_row = []
            for col in range(cols):
                block_x = col * width
                block_y = row * height
                rect = pygame.Rect(block_x, block_y, width, height)
                block_row.append(rect)
                block_exist_row.append(True)
            self.blocks.append(block_row)
            self.blocks_exist.append(block_exist_row)


class Paddle():
    def __init__(self, rows, cols, screen_width, screen_height):
        self.screen_width = screen_width
        self.height = int(screen_height * 0.05)
        self.width = int(screen_width / cols)
        self.x = int(screen_width / 2 - self.width / 2)
        self.y = screen_height - (self.height * 2)
        self.speed = int(screen_width / 60) * 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, action):
        if 2 > action < 0:
            print(f"ERROR IN BREAKOUT ENVIRONMENT: received action {action}")
            exit()
        if action == 1 and self.rect.left > 0:
            self.rect.x -= self.speed
        if action == 2 and self.rect.right < self.screen_width:
            self.rect.x += self.speed


class Ball:
    def __init__(self, start_x, start_y, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = int(screen_width * 0.015)
        self.x = start_x - self.radius
        self.y = start_y
        self.rect = pygame.Rect(self.x, self.y, self.radius * 2, self.radius * 2)
        self.speed_x = 4 * 2
        self.speed_y = -4 * 2
        self.speed_max = 5 * 2

    def move(self, blocks, blocks_exist, paddle):
        collision_thresh = 5 * 2

        reward = 0

        for row_id, row in enumerate(blocks):
            for block_id, block in enumerate(row):
                if not blocks_exist[row_id][block_id]:
                    continue
                if self.rect.colliderect(block):
                    # collision from above
                    if abs(self.rect.bottom - block.top) < collision_thresh and self.speed_y > 0:
                        self.speed_y *= -1
                    # collision from bellow
                    if abs(self.rect.top - block.bottom) < collision_thresh and self.speed_y < 0:
                        self.speed_y *= -1
                    # collision from left
                    if abs(self.rect.right - block.left) < collision_thresh and self.speed_x > 0:
                        self.speed_x *= -1
                    # collision from right
                    if abs(self.rect.left - block.right) < collision_thresh and self.speed_x < 0:
                        self.speed_x *= -1
                    blocks_exist[row_id][block_id] = False
                    # del wall.blocks[row_id][block_id]
                    reward += 1

        done_ = True
        for row in blocks_exist:
            for block_exists in row:
                if block_exists:
                    done_ = False
                    break
        if done_:
            reward = 10
        done = done_

        # Check collision on wall
        if self.rect.left < 0 or self.rect.right > self.screen_width:
            self.speed_x *= -1
        if self.rect.top < 0:
            self.speed_y *= -1

        # Check bottom of screen
        if self.rect.bottom > self.screen_height:
            done = True
            reward -= 10

        # Check paddle
        if self.rect.colliderect(paddle):
            if abs(self.rect.bottom - paddle.rect.top) < collision_thresh and self.speed_y > 0:
                self.speed_y *= -1
            else:
                self.speed_x *= -1

        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        return reward, done, blocks, blocks_exist


class Game():
    def __init__(self):
        pygame.init()

        self.done = False

        self.screen_width = 600  # todo change later
        self.screen_height = 600  # todo change later
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Breakout")

        self.bg = (234, 218, 184)
        self.block_color = (242, 85, 96)
        self.paddle_color = (142, 135, 123)
        self.paddle_outline = (100, 100, 100)

        self.rows = 6
        self.cols = 6

        self.n_actions = 3
        self.n_states_disc = 10 * 10 * 2 * 2 * 10

    def reset(self):
        # initialize wall, ball, paddle
        self.wall = Wall(self.rows, self.cols, self.screen_width, self.screen_height)
        self.paddle = Paddle(self.rows, self.cols, self.screen_width, self.screen_height)
        self.ball = Ball(randrange(20, 560),
                         randrange(380, 520), self.screen_width,
                         self.screen_height)  # 380
        # return state
        return self.get_current_state()

    def get_current_state(self):
        # for image: return self.screen
        ball_x = self.ball.rect.x
        ball_y = self.ball.rect.y
        abs_speed = abs(self.ball.speed_x)
        ball_speed_x = self.ball.speed_x / abs_speed
        ball_speed_y = self.ball.speed_y / abs_speed
        paddle_x = self.paddle.rect.x + self.paddle.width / 2
        return self.discretize_state([ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_x])

    def step(self, action):
        # move paddle
        self.paddle.move(action)

        # move ball and potentially break the bricks
        reward, done, blocks, blocks_exist = self.ball.move(self.wall.blocks, self.wall.blocks_exist, self.paddle)
        self.wall.blocks_exist = blocks_exist
        self.wall.blocks = blocks

        state = self.get_current_state()

        # Feedback
        ball_x = self.ball.rect.x
        feedback = 1 if self.paddle.rect.x <= ball_x < self.paddle.rect.x + self.paddle.width else -1

        return state, reward, done, feedback

    def draw(self):
        self.screen.fill(self.bg)
        self.draw_wall()
        self.draw_paddle()
        self.draw_ball()
        pygame.display.update()

    def draw_wall(self):
        for row_id, row in enumerate(self.wall.blocks):
            for b_id, block in enumerate(row):
                if not self.wall.blocks_exist[row_id][b_id]:
                    continue
                pygame.draw.rect(self.screen, self.block_color, block)
                pygame.draw.rect(self.screen, self.bg, (block), 2)

    def draw_paddle(self):
        pygame.draw.rect(self.screen, self.paddle_color, self.paddle.rect)
        pygame.draw.rect(self.screen, self.paddle_outline, self.paddle.rect, 2)

    def draw_ball(self):
        pygame.draw.circle(self.screen, self.paddle_color,
                           (self.ball.rect.x + self.ball.radius, self.ball.rect.y + self.ball.radius), self.ball.radius)
        pygame.draw.circle(self.screen, self.paddle_outline,
                           (self.ball.rect.x + self.ball.radius, self.ball.rect.y + self.ball.radius), self.ball.radius,
                           2)

    def discretize_state(self, state):
        ball_x = 9 if state[0] >= 600 else int(state[0] / 60)
        ball_y = 9 if state[1] >= 600 else int(state[1] / 60)
        ball_speed_x = 0 if state[2] == -1 else 1
        ball_speed_y = 0 if state[3] == -1 else 1
        paddle_x = 9 if state[4] >= 600 else int(state[4] / 60)
        return [ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_x]


"""
# Example
env = Game()
for episode in range(10):
    state = env.reset()

    for step in range(100):
        # action = np.random.choice([0, 1, 2])
        action = 2
        new_state, reward, done, feedback = env.step(action)

        env.draw()
        time.sleep(0.1)
        print(new_state, ",", reward, ",", done)

        if done == True:
            break
"""
