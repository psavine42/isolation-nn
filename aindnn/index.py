"""
pseudo code

table
Feature     Policy      Value


optimization
    pi_t  <- p_t
    z     <- v_t




index
---game---
a -> action
s -> state

---nn---
theta -> network parameters

f -> network function

p -> vector of probabilites of each move given s

v -> evaluation of a position. E(-1 , 1).
    probability of position being winning

eq.1 (p, v) = f_theta(s)
    standard nn evaluation. network outputs p and v from
    respective heads

eta <-
dir <- Dirichlet noise on prior a

dir = 0.03
eta = 0.25
resignation_threshold
win threshold


P(s, a) = (1 - eta) * p_a + eta * Dir(0.03, p_a)

---mcts---


EDGE (s, a)

eq2.    argmax Q(s, a) + U(s, a)
eq3.1   U(s, a) ~= P(s, a) / (1 + N(s, a))
eq3.2.  U(s, a) = c_puct * P(s, a) * (Sum_b N(s, b) / (1 + N(s, a)))
eq4.    Q(s,a) = 1 / N(s,a) * (sum s' | s,a -> s' * V(s'))


pi ->  pi( a | s0) = (N( s0, a) ^ 1/tao) / sum_b N(s0, b)^1/tao


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


---controls---
tao -> temperature of MCTS search
N -> number of visits from root state


N ^ 1/tao

"""



class Network:
    """
    (p, v) = f_theta(s)
    policy, value = network theta of state

    p [policy]
        -is a vector of move porbabilites for selecting each move (p_a)
        p_a = Pr ( a | s ) probability of action given stat (s)


    """
    def execute_mstc(self, position):
        """
        returns mstc probablities (pi) of playing a move
            (dm-note: these return stronger moves than the raw
                probabilitis 'p' returned by f_theta(s)
                MSTC can be seen as POLICY IMPROVEMENT OPERATOR)

        :recieves grad = False (?)

        :param position:
        :return: pi
        """
        pass

    def PSEUDO_backward(self):
        """
        update parameters theta to make move probabilities (p, v) = f_theta(S)
        more closely match the improved search probabilities and
        self play closer to self play winner (pi, z).
        parameters f_theta^t+1 will theoretically be stronger
        :return:
        """
        pass


    def forward(self, input):
        """one network, no Monte Carlo roll out at train time

        :recieves grad = True (?)
        """



        pass












