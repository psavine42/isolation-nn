

## Synopsis

Working on a alphago like solution to the isolation problem. Because its fun to implement papers. More results when I finish the latest batch of generation and minimax figuring out.

Instructions on running forthcoming. Trained Networks should be available at https://www.floydhub.com/psavine/projects/isolation/6/code/outputx/goodpool.

Alphago architecture vs this architecture:
Network architecture (Policy baseline):                     
features: k = 64 (also tried 16, but unstable)                   
Layer 1:  Convolution with k filters, size 3, stride 1, pad 2, relu    
Layer 2-4:  Convolution with k filters, size 3, stride 1, relu         
Layer 5: Convolution with 1 filter, size 1, stride 1, bias, relu       
Value Networks addition (goal is to get one answer of position value):
Layer 6: fully connected layer with 128 units                                
Layer 7: fully connected layer with 1 tanh

AlphaGo Network architecture
features: k=128 (also with 256, 384)
Layer 1: Convolution w/ k filters, stride 5, pad 2, relu
Layer2-12: Convolution w/ k filters, size 3, stride 1, relu
Layer 13: Convolution 1 filter, 1 3, stride 1, bias, softmax
Value Networks addition:
Layer 14: fully connected layer, 256 units
Layer 15: fully Connected Layer with 1 tanh.
** my layer 5 is a relu which was an oversight on my part, which I realized too late,so need to retrain. **
alpahgo paper https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
Other than that, its a ctrl+c, ctrl+v.

Example of position in isolation

         0   1   2   3   4   5   6
    0  |   |   |   |   |   |   |   |
    1  |   |   |   |   | 1 |   |   | 
    2  |   | 2 |   |   | - | - |   | 
    3  |   |   | - |   | - | - |   | 
    4  |   |   | - | - |   |   | - | 
    5  |   | - |   |   |   |   |   | 
    6  |   |   |   |   |   |   |   |

And here is how it is fed into the convnet:

    Player Position      Opponent position  Open moves          All ones       
    [[[0 0 0 0 0 0 0]    [[0 0 0 0 0 0 0]  [[1 1 1 1 1 1 1]   [[1 1 1 1 1 1 1]
      [0 0 0 0 0 0 0]     [0 0 0 0 1 0 0]   [1 1 1 1 0 1 1]    [1 1 1 1 1 1 1]
      [0 1 0 0 0 0 0]     [0 0 0 0 0 0 0]   [1 0 1 1 0 0 1]    [1 1 1 1 1 1 1]
      [0 0 0 0 0 0 0]     [0 0 0 0 0 0 0]   [1 1 0 1 0 0 1]    [1 1 1 1 1 1 1]
      [0 0 0 0 0 0 0]     [0 0 0 0 0 0 0]   [1 1 0 0 1 1 0]    [1 1 1 1 1 1 1]
      [0 0 0 0 0 0 0]     [0 0 0 0 0 0 0]   [1 0 1 1 1 1 1]    [1 1 1 1 1 1 1]
      [0 0 0 0 0 0 0]]    [0 0 0 0 0 0 0]]  [1 1 1 1 1 1 1]]   [1 1 1 1 1 1 1]]
      
      Legal moves         Closed move       All zeros
     [[1 0 1 0 0 0 0]    [[0 0 0 0 0 0 0]  [[0 0 0 0 0 0 0]
      [0 0 0 1 0 0 0]     [0 0 0 0 0 0 0]   [0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0]     [0 0 0 0 1 1 0]   [0 0 0 0 0 0 0]
      [0 0 0 1 0 0 0]     [0 0 1 0 1 1 0]   [0 0 0 0 0 0 0]
      [1 0 0 0 0 0 0]     [0 0 1 1 0 0 1]   [0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0]     [0 1 0 0 0 0 0]   [0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0]]    [0 0 0 0 0 0 0]]  [0 0 0 0 0 0 0]]]


Udacity imposes a 150ms limit per move, and with moving stuff up and down from gpu and my not planning anything, this kills performance, but I will figure out the solution. Without time limit, this beats up on pure tree search, but at 150ms, it gets killed. When running tests with relaxed time constraints, neural net wins almost every time (90%), even with currently bad implementation. So basically, the capacity is there for the network to learn the board and heuristics, and if I get to finish the move selection bit and do some retraining, it should do significantly better.

                          150ms/mv        1000ms/mv
    Match #   Opponent    AB_NN           AB_NN
                          Won | Lost      Won | Lost
      1       Random       9  |   1       10  |   0  
      2       MM_Open      7  |   3        8  |   2  
      3      MM_Center     9  |   1        9  |   1  
      4     MM_Improved    5  |   5        9  |   1      
      5       AB_Open      2  |   8        6  |   4       
      6      AB_Center     1  |   9        6  |   4      
      7     AB_Improved    1  |   9        5  |   5      
    --------------------------------------------------
           Win Rate:      48.5%            75.7%
           
Files:
controller.py   -training, game generation
nn_players.py   -neural network game agent
game_agents.py  -the base code for udacity lesson (+some benchmarking)
tournament.py   -run a tournament among agents (heavily modified from lesson code)
aindnn/slnn.py  -neural network definitions
dataloaders.py  -pytorch subclass loader for training data
laoders.py      -to clean up.


# TODO

-If you can help  with reinforcment learning part (the big chunk of comments in controller.py), that is much appreciated
-implement pure numpy version (no pytorch). I have base code there, but totally lifted from https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
-working on strategies for move selection as implementend in alphago.


# Build a Game-playing Agent [udacity lesson]

![Example game of isolation](viz.gif)

## Synopsis

In this project, students will develop an adversarial search agent to play the game "Isolation".  Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.  Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.  These rules are implemented in the `isolation.Board` class provided in the repository. 

This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).  The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).

Additionally, agents will have a fixed time limit each turn to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the opponent wins.

Students only need to modify code in the `game_agent.py` file to complete the project.  Additional files include example Player and evaluation functions, the game board class, and a template to develop local unit tests.  



