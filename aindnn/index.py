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


P(s ,a _ = (1 - eta) * p_a + eta * Dir(0.03, p_a)

---mcts---


EDGE (s, a)

eq2. argmax Q(s, a) + U(s, a)
eq3. U(s, a) ~= P(s, a) / (1 + N(s, a))

pi -> search probablities computed by todo

---controls---
tao -> temperature of MCTS search
N -> number of visits from root state


N ^ 1/tao

"""

"""
Node - {:state, parent}


"""














