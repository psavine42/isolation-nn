import numpy as np


def game_to_input(game):
    x, y = game.height, game.width
    L_pos_self, L_zeros, L_pos_opp, L_legal, L_open, = \
        [np.zeros((x, y), dtype=int) for _ in range(5)]
    L_ones, L_closed = [np.ones((x, y), dtype=int) for _ in range(2)]

    for row, col in game.get_legal_moves():
        L_legal[row][col] = 1

    for row, col in game.get_blank_spaces():
        L_open[row][col] = 1

    s_r, s_c = game.get_player_location(game.active_player)
    L_pos_self[s_r][s_c] = 1

    o_r, o_c = game.get_player_location(game.inactive_player)
    L_pos_opp[o_r][o_c] = 1

    L_closed = L_closed - L_open - L_pos_opp - L_pos_self
    return np.stack([L_pos_self, L_pos_opp, L_open, L_ones, L_legal, L_closed, L_zeros], axis=0)
