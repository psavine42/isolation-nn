import numpy as np
import isolation as iso
from agents import game_agent as ga
import os


_history_ = 'hist'
_board_size_ = 'size'
_duration_ = 'duration'

defkeys = [_history_, _board_size_, _duration_]


def read_npz(filename, names):
    npz = np.load(filename)
    ret = dict((name, npz[name].copy()) for name in names)
    npz.close()
    return ret


def one_hot_move_to_index(w, move_index):
    r = move_index // w
    c = move_index - (r * w)
    return r, c


def append_pair_to_stack(move_pair, idx, positions_stack, game, last_self_pos, last_opp_pos):
    """
        Initially my gpu was taking lik 20 secs to do this,
        so I made updates to stack as fast as i could...
        turns out this was pointless.
        update self fast index to create game inputs:
            0 L_pos_self [0 -> 1], prev [1 -> 0] xx
            1 L_pos_opp  [0 -> 1], prev [1 -> 0] xx
            2 L_open     [1 -> 0], [1 -> 0] xx
            3 L_ones
            4 L_legal    clear->0, get->1 xx
            5 L_closed   [0->1],  [0->1] xx
            6 L_zeros
        """
    # for opp_move, own_move
    pos_self, pos_opp  = move_pair

    positions_stack[idx][4][::] = 0           # L_legal
    for row, col in game.get_legal_moves():
        positions_stack[idx][4][row][col] = 1 # L_legal

    if pos_self:
        s_r, s_c = pos_self
        positions_stack[idx][0][::] = 0       # L_pos_self
        positions_stack[idx][0][s_r][s_c] = 1 # L_pos_self

    if last_opp_pos:
        lor, loc = last_opp_pos
        positions_stack[idx][5][lor][loc] = 1 # L_closed
        positions_stack[idx][2][lor][loc] = 0 # L_open
        positions_stack[idx][1][lor][loc] = 0 # L_pos_opp

    if last_self_pos:
        ro, co = last_self_pos
        positions_stack[idx][5][ro][co] = 1 # L_closed
        positions_stack[idx][2][ro][co] = 0 # L_open

    # pos_op = game.get_player_location(game.get_opponent(self))
    if pos_opp:
        positions_stack[idx][1][pos_opp[0]][pos_opp[1]] = 1 # L_pos_opp

    return positions_stack


def process_one_file(file_dict, player=0, to_file=False, stop_at=100):
    positions, moves = [], []
    hist = file_dict[_history_]
    player1, player2 = ga.IsolationPlayer(), ga.IsolationPlayer()
    game = iso.Board(player1, player2)

    # print("------------------------------------")
    # layers for ones and zeros fill

    x, y = file_dict[_board_size_]
    L_zeros, L_ones, L_open, L_closed = [np.zeros((x, y), dtype=int) for _ in range(4)]

    L_ones.fill(1)
    L_open.fill(1)

    # replay positions
    for m_idx in range(len(file_dict[_history_])):

        move_r, move_c = hist[m_idx]

        if move_r == -1 and move_c == -1 or m_idx >= stop_at:
            break

        if m_idx > 1:
            prev_move_r, prev_move_c = hist[m_idx - 2]
            L_open[prev_move_r][prev_move_c] = 0

        if m_idx > 2:
            prev_move2_r, prev_move2_c = hist[m_idx - 3]
            L_open[prev_move2_r][prev_move2_c] = 0
            L_closed[prev_move2_r][prev_move2_c] = 1

        if (m_idx + player) % 2 == 0:
            # player 1's turn
            L_pos_self, L_pos_opp, L_legal, L_move = [np.zeros((x, y), dtype=int) for _ in range(4)]
            L_move[move_r][move_c] = 1

            for row, col in game.get_legal_moves():
                L_legal[row][col] = 1

            if m_idx > 0:
                prev_opp_move_r, prev_opp_move_c = hist[m_idx - 1]
                L_pos_opp[prev_opp_move_r][prev_opp_move_c] = 1

            if m_idx > 1:
                L_pos_self[prev_move_r][prev_move_c] = 1

            inputs = np.stack([L_pos_self, L_pos_opp, np.copy(L_open),
                               np.copy(L_ones), L_legal, np.copy(L_closed),
                               np.copy(L_zeros)], axis=0)

            # not one hot encoding, but index of move, cuz pytorch
            moves.append(np.where(L_move.flatten() == 1)[0])
            positions.append(inputs)
        game = game.forecast_move((move_r, move_c))
    return np.asarray(positions), np.asarray(moves) 


def parse_log(path):
    f = open(path, 'r')
    steps, train_loss, val_loss, accuracy, step = [], [], [], [], False

    while True:
        text = f.readline()
        if len(text) > 0:
            if text[0] == 'a':
                step = True
                cols = text.split(',')
                accuracy.append(float(cols[0].split(': ')[1]))
                val_loss.append(float(cols[1].split(': ')[1]))
                
            if text[0] == 'E' and step:
                step = False
                train_loss.append(float(cols[2].split(': ')[1]))
                steps.append(float(cols[1].split(': ')[1]))
    f.close()
    return train_loss, val_loss, accuracy, steps


def log_graph(path):
    tl, vl, acc, st = parse_log(path)
    plt.plot(tl, vl, acc)
    plt.ylabel('some numbers')
    plt.show()


def normalize_dataset(dir_in, dir_out):
    for filename in os.listdir(dir_in):
        fd = read_npz(dir_in + filename, defkeys)
        
        p1, m1 = process_one_file(fd, player=0, to_file=True)
        p2, m2 = process_one_file(fd, player=1, to_file=True)

        fln = os.path.splitext(filename)[0]
        for idx in range(len(p1)):
            np.savez_compressed(dir_out + fln + 'm' + str(idx) + 'p1.npz',
                                 position=p1[idx], move=m1[idx])

        for idx2 in range(len(p2)):
            np.savez_compressed(dir_out + fln + 'm' + str(idx2) + 'p2.npz',
                                 position=p2[idx2], move=m2[idx2])




     