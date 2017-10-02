import numpy as np
import isolation as iso
import game_agent as ga
import torch
from torch.utils import data
import glob
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

def process_one_file(file_dict, player=0, to_file=False):
    #L_zeros, L_ones, L_move, L_pos_self, L_pos_opp, L_legal, L_open, L_closed
    #L_self
    positions = []
    moves = []

    hist = file_dict[_history_]
    player1, player2 = ga.IsolationPlayer(), ga.IsolationPlayer()
    game = iso.Board(player1, player2)
    #print("------------------------------------")
    #layers for ones and zeros fill
    x, y = file_dict[_board_size_]
    L_zeros, L_ones, L_open, L_closed = [np.zeros((x, y), dtype=int) for _ in range(4)]

    L_ones.fill(1)
    L_open.fill(1)

    #replay positions
    for m_idx in range(len(file_dict[_history_])):
        move_r, move_c = hist[m_idx]

        if move_r == -1 and move_c == -1:
            break

        if m_idx > 1:
            prev_move_r, prev_move_c = hist[m_idx - 2]
            L_open[prev_move_r][prev_move_c] = 0

        if m_idx > 2:
            prev_move2_r, prev_move2_c = hist[m_idx - 3]
            L_open[prev_move2_r][prev_move2_c] = 0
            L_closed[prev_move2_r][prev_move2_c] = 1

        if (m_idx + player) % 2 == 0:

            #player 1's turn
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
                                np.copy(L_ones), L_legal, np.copy(L_closed), np.copy(L_zeros)], axis=0)

            #not one hot encoding, but index of move, cuz pytorch
            moves.append(np.where(L_move.flatten()==1)[0])
            # print(np.where(L_move.flatten()==1)[0])
            # print(L_move)
            #print(inputs)
            #print(player, move_r, move_c, np.where(L_move.flatten()==1)[0])
            positions.append(inputs)

        game = game.forecast_move((move_r, move_c))

    return np.asarray(positions), np.asarray(moves) 




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



def normalize_dataset(dir_in, dir_out):

    for idx in range(len(p1)):
        np.savez_compressed(dir_out + fln + 'm' + str(idx) + 'p1.npz',
                            position=p1[idx], move=m1[idx])

     