import argparse
import aindnn.MCTS as mcts

class Env(object):
    def __init__(self, args):
        self.args = args # save a copy of args


def PUCT(node, c):
    """
     c_puct is a constant which initially prefers high prob actions
     with low visit count, but asymptotically prefers actions with high-value
    :return:
    """

# todo game and state thing
"""
# env
#   -> creates Agent
#       -> Agent creates NNs
#
#   -> sets up MTCS
#   -> sets up Board
# -> process
#   -> optimize θ
#   -> evaluate player aθ_i
#   -> generate new data with aθ*

"""
"""
figure a-c

2a - Select
    start at node, search until t or L
    each search uses PUCT

2b - Forward - Expand and evaluate
    add leaf node to queue for NN evaluation
    (p, v) <- evaluate + ?? dihedral reflection
    Leaf node is expanded
    Each edge is initialized
    backup value
    
2c - Backup (v)
    increment count
    Action Value updated 

2d - Play 
    t <- temperature
    a <- select max move(s , t)
    ST <- update Tree with Action a
     
"""
"""

"""


# AlphaGo Zero Node
# N(
# N(s, a) -> num visits
# P(s, a) -> probability
# Q(s, a) ->


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("echo")
    parser.add_argument('--q', nargs='?', type=int, default=0, help='[]')
    args = parser.parse_args()
    print(args.echo)

