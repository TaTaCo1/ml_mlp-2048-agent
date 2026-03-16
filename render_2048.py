import pygame
import numpy as np

# action names for display
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

class Game2048Renderer:
    def __init__(self, size=5, tile_size=100):
        self.size = size
        self.tile_size = tile_size
        self.board_size = size * tile_size
        self.header_size = 80
        self.window_height = self.board_size + self.header_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.board_size, self.window_height))
        pygame.display.set_caption("2048 - RL Environment")
        self.font = pygame.font.Font(None, 40)
        self.score_font = pygame.font.Font(None, 50)
        self.move_font = pygame.font.SysFont("freesans", 45)
        self.colors = {
            0: (205, 193, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }

    # FIX: added last_action parameter
    def render(self, board, score, last_action=None):
        board = self.decode_board(board)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((250, 248, 239))

        # draw score on the left
        score_text = self.score_font.render(f"Score: {score}", True, (119, 110, 101))
        self.screen.blit(score_text, (10, 10))

        # draw last move on the right if provided
        if last_action is not None:
            arrow = ACTION_ARROWS.get(last_action, "?")
            name = ACTION_NAMES.get(last_action, "?")
            move_text = self.move_font.render(f"{arrow}", True, (119, 110, 101))
            # align to right side of header
            move_rect = move_text.get_rect()
            move_rect.right = self.board_size - 10
            move_rect.top = 20
            self.screen.blit(move_text, move_rect)

        pygame.draw.rect(
            self.screen,
            (187, 173, 160),
            (0, self.header_size, self.board_size, self.board_size),
        )

        # draw tiles
        for r in range(self.size):
            for c in range(self.size):
                value = board[r][c]
                color = self.colors.get(value, (60, 58, 50))
                rect = pygame.Rect(
                    c * self.tile_size,
                    self.header_size + r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, color, rect)
                if value != 0:
                    text = self.font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        pygame.display.flip()

    def decode_board(self, obs):
        powers = np.argmax(obs, axis=2)
        board = np.power(2, powers)
        board[board == 1] = 0
        return board