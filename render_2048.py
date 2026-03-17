import pygame
import numpy as np

# action names for display
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

TILE_COLORS = {
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


def decode_board(obs):
    """Shared decode logic — used by both renderers."""
    powers = np.argmax(obs, axis=2)
    board = np.power(2, powers)
    board[board == 1] = 0
    return board


def draw_board(screen, board, score, last_action,
               offset_x, offset_y, size, tile_size,
               header_size, fonts, label=None):
    """
    Shared board drawing logic.
    Used by both single and side by side renderers.
    """
    font, score_font, move_font, label_font = fonts

    # header background
    pygame.draw.rect(screen, (250, 248, 239),
                 (offset_x, offset_y, size * tile_size, header_size))

    # label if provided
    if label:
        label_text = label_font.render(label, True, (119, 110, 101))
        screen.blit(label_text, (offset_x + 8, offset_y + 8))

    # score
    score_text = score_font.render(f"Score: {score}", True, (119, 110, 101))
    screen.blit(score_text, (offset_x + 10, offset_y + 40))

    # last action arrow
    if last_action is not None:
        arrow = ACTION_ARROWS.get(last_action, "?")
        move_text = move_font.render(arrow, True, (119, 110, 101))
        move_rect = move_text.get_rect()
        move_rect.right = offset_x + size * tile_size - 10
        move_rect.top = offset_y + 20
        screen.blit(move_text, move_rect)

    # board background
    board_y = offset_y + header_size
    pygame.draw.rect(screen, (187, 173, 160),
                     (offset_x, board_y, size * tile_size, size * tile_size))

    # tiles
    for r in range(size):
        for c in range(size):
            value = board[r][c]
            color = TILE_COLORS.get(value, (60, 58, 50))
            rect = pygame.Rect(
                offset_x + c * tile_size + 4,
                board_y + r * tile_size + 4,
                tile_size - 8,
                tile_size - 8
            )
            pygame.draw.rect(screen, color, rect, border_radius=6)
            if value != 0:
                text = font.render(str(value), True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)


class Game2048Renderer:
    """Single board renderer."""

    def __init__(self, size=4, tile_size=100):
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
        self.label_font = pygame.font.Font(None, 30)
        self.fonts = (self.font, self.score_font, self.move_font, self.label_font)

    def render(self, board, score, last_action=None):
        board = decode_board(board)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((250, 248, 239))

        draw_board(self.screen, board, score, last_action,
                   offset_x=0, offset_y=0,
                   size=self.size, tile_size=self.tile_size,
                   header_size=self.header_size,
                   fonts=self.fonts)

        pygame.display.flip()

    def decode_board(self, obs):
        return decode_board(obs)


class Game2048TripleRenderer:
    """Three boards side by side — DQN | DQN+MCTS | Random."""

    GAP = 20

    def __init__(self, size=4, tile_size=100):
        self.size = size
        self.tile_size = tile_size
        self.board_size = size * tile_size
        self.header_size = 80
        self.window_width = self.board_size * 3 + self.GAP * 2
        self.window_height = self.board_size + self.header_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048 — Random | DQN | DQN+MCTS")
        self.font = pygame.font.Font(None, 40)
        self.score_font = pygame.font.Font(None, 36)
        self.move_font = pygame.font.SysFont("freesans", 36)
        self.label_font = pygame.font.Font(None, 30)
        self.fonts = (self.font, self.score_font, self.move_font, self.label_font)

    def render(self,
               obs_left,   score_left,   action_left,   label_left,
               obs_middle, score_middle, action_middle, label_middle,
               obs_right,  score_right,  action_right,  label_right):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((250, 248, 239))


        # right board — Random
        draw_board(self.screen, decode_board(obs_right),
                   score_right, action_right,
                   offset_x=(self.board_size + self.GAP) * 2, offset_y=0,
                   size=self.size, tile_size=self.tile_size,
                   header_size=self.header_size,
                   fonts=self.fonts, label=label_right)
        
        # left board — DQN
        draw_board(self.screen, decode_board(obs_left),
                   score_left, action_left,
                   offset_x=0, offset_y=0,
                   size=self.size, tile_size=self.tile_size,
                   header_size=self.header_size,
                   fonts=self.fonts, label=label_left)

        # middle board — DQN+MCTS
        draw_board(self.screen, decode_board(obs_middle),
                   score_middle, action_middle,
                   offset_x=self.board_size + self.GAP, offset_y=0,
                   size=self.size, tile_size=self.tile_size,
                   header_size=self.header_size,
                   fonts=self.fonts, label=label_middle)

        pygame.display.flip()