from multiprocessing import Process
from aindnn.MCTS.nodes import Node
from aindnn.Trainers.trainers import Trainer, MiniBatch


class Evaluator(Trainer, Process):
    """
    Play match between two Agents. save winner
    emit current best to any process requiring best agent

    Attributes
        threshold (float):
        tao (float): temperature for exploration
    """
    def __init__(self,
                 env,
                 args,
                 in_queue=None,
                 out_queue=None,
                 tao=1e-4,
                 win_threshold=0.55,
                 num_games=400,
                 num_sims=1000):
        super(Evaluator, self).__init__(args)
        Process.__init__(self)
        # common
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.env = env

        # class specific
        self.tao = tao
        self.threshold = win_threshold
        self.num_games = num_games
        self.num_sims = num_sims
        self.on_init()

    def on_init(self):
        if self.args.best_model:
            self.load_agent(self.args.best_model)
            self.on_new_agent()

    def on_new_agent(self):
        self.agent.num_sims = self.num_sims

    def reset(self):
        self.agent = None

    def send_off(self):
        self.out_queue.put(MiniBatch('agent', self.agent, None))
        return True

    def play_game(self, challenger):
        """
        call network.forward until game is complete.
        when game is complete, return result z

        Need a place to put the mcts and states

        :return: z - game result
        """
        turn, game = 0, Node(self.env(), None)
        # game.terminal -> inactive wins
        # play until terminal is reached
        while not game.terminal:
            if turn == 1:
                prob, move = challenger.forward(game, self.tao)
            else:
                prob, move = self.agent.forward(game, self.tao)
            game = game.forward(move)
            turn ^= 1           # flip 'active' bit
        return turn

    def forward(self, challenger):
        wins = 0
        for _ in range(self.num_games):
            wins += self.play_game(challenger)
        if wins / self.num_games > self.threshold:
            self.agent = challenger
            return self.send_off()
        return True

    def run(self):
        while True:
            next_task = self.in_queue.get()
            self.log('EVAL - MESSAGE RECEIVED', next_task)

            if next_task is None:
                print('%s: Exiting' % self.name)
                self.in_queue.task_done()
                break
            act, agent, _ = next_task()
            if act == 'agent':
                agent.num_sums = self.num_sims
                if self.agent is None:
                    self.agent = agent
                else:
                    self.forward(agent)
            self.in_queue.task_done()
        return



