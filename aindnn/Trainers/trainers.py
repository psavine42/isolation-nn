import torch, os
from torch.utils.data import DataLoader, Dataset
from aindnn.MCTS.nodes import Node
from torch.autograd import Variable
from multiprocessing import Process


class MiniBatch(object):
    def __init__(self, s, pi, z):
        self.s = s
        self.pi = pi
        self.z = z

    def __call__(self):
        return self.s, self.pi, self.z

    def __str__(self):
        return '{} {} {}'.format(self.s, self.pi, self.z)


class Trainer(object):
    """
    subclass for training processes
    It should handle:
        -Plotting
        -logging
        -

    """
    def __init__(self,
                 args,
                 plots=None):
        """

        :param args:
        :param plots:
        """
        super(Trainer, self).__init__()
        self.args = args
        self.agent = None
        self.verbose = args.verbose
        self.show_every = args.show_every
        self.viz = args.viz
        self._plots = {}

    def log(self, *args):
        if self.verbose is True:
            print(args)

    def plog(self, *args):
        print(*args)

    @staticmethod
    def to_vis(tnsr):
        if isinstance(Variable, tnsr):
            return tnsr.data.cpu().numpy()
        else:
            return tnsr.cpu().numpy()

    def load_agent(self, path):
        if path is not None and os.path.exists(path):
            self.agent = torch.load(path)

    def save_agent(self, path):
        if path is not None and os.path.exists(path):
            torch.save(self.agent, path)

    def visualize(self, key, data, step):
        self.viz.line(X=torch.ones((1, len(data))).cpu() *step,
                      Y=torch.Tensor(data).unsqueeze(0).cpu(),
                      win=self._plots[key],
                      update='append')
        return

    def reset(self, **new_data):
        pass

    def forward(self, inputs):
        pass


class SupervisedZero(Trainer):
    """
    Loader handles the Game Object

    """
    def __init__(self,
                 loader_Klass,
                 data_dir,
                 args,
                 queue=None,
                 network=None,
                 batch_size=8,
                 n_epochs=25000):
        super(SupervisedZero, self).__init__(args=args)
        self.agent = network
        self.task_queue = queue
        self.batch_size = batch_size
        self.loader_Klass = loader_Klass
        self.n_epochs = n_epochs
        self.data_dir = data_dir
        # CoupledLoader
        self.train_data = loader_Klass(self.data_dir, train=True, vl=True, pct_train=0.95)
        self.valid_data = loader_Klass(self.data_dir, train=False, vl=True, pct_train=0.95)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, shuffle=args.shuffle)
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=1, shuffle=True)


    def retrieve(self):
        return self.agent, self.cache

    def reset(self):
        self.agent = None
        self.cache = []

    def batchify(self, s, pi, z):
        """
        Convert numpy outputs of self play game
        :param s:
        :param pi:
        :param z:
        :return:
        """
        batches = []
        s_c = s.split(self.batch_size, 0)
        pi_c = pi.split(self.batch_size, 0)
        z_c = z.split(self.batch_size, 0)
        self.log('chunks', z.size(0), len(s_c))

        for i in range(len(s_c)):
            mb = MiniBatch(s_c[i], pi_c[i], z_c[i])
            if self.task_queue is not None:
                self.task_queue.put(mb)
            else:
                batches.append(mb)
        self.task_queue.join()
        return batches

    def play_game(self, agent):
        """
        call network.forward until game is complete.
        when game is complete, return result z

        Need a place to put the mcts and states

        :return: z - game result
        """
        turn, move, states, pis, zs = 0, 0, [], [], []
        # s_node = Node(self.env(), None)

        even_won = 1 if turn == 0 else -1
        for idx in range(move):
            z = even_won if idx % 2 == 0 else -1 * even_won
            zs.append(torch.FloatTensor([z]))

        # print(s_node._state.to_string())
        self.log('winner: ', turn + 1)
        mbs = self.batchify(torch.cat(states, 0), torch.stack(pis, 0), torch.stack(zs, 0))
        return mbs

    def forward(self, __na__):
        batches = []
        for idx in range(self.n_epochs):
            pass
        return batches


class SelfPlay(Trainer, Process):
    """

    lookahead search is inside the training loop

    Attributes
        game (Game Object):
            must implement .score(), .terminal(),
        num_sims (int) :
        num_games (int) :

    """
    def __init__(self,
                 env,
                 args,
                 in_queue=None,
                 out_queue=None,
                 num_sims=1600,
                 num_games=25000):
        super(SelfPlay, self).__init__(args)
        Process.__init__(self)

        self.out_queue = out_queue
        self.in_queue = in_queue
        #
        self.batch_size = args.batch_size
        self.false_pos_resignations = 0

        self.resign_threshold = -0.9
        self.num_games = num_games
        self.num_sims = num_sims
        self.env = env

    def retrieve(self):
        return self.agent

    def reset(self):
        self.agent = None

    def send_off(self, s, pi, z):
        """
        Convert numpy outputs of self play game
        :param s:
        :param pi:
        :param z:
        :return:
        """
        batches = []
        s_c = s.split(self.batch_size, 0)
        pi_c = pi.split(self.batch_size, 0)
        z_c = z.split(self.batch_size, 0)
        self.log('chunks', z.size(0), len(s_c))

        for i in range(len(s_c)):
            mb = MiniBatch(s_c[i], pi_c[i], z_c[i])
            if self.out_queue is not None:
                self.out_queue.put(mb)
                self.out_queue.join()
            else:
                batches.append(mb)

        return batches

    def play_game(self, agent, can_resign=True):
        """
        call network.forward until game is complete.
        when game is complete, return result z

        Need a place to put the mcts and states

        :return: z - game result
        """
        turn, move, states, pis, zs = 0, 0, [], [], []
        s_node = Node(self.env(), None)

        while not s_node.terminal:
            tao = 1 if move < 5 else 10e-3
            final_action, pi = agent.forward(s_node, tao=tao)
            if final_action is None:
                break

            # add state and value data
            s_node = final_action.child
            states.append(s_node.to_input())
            pis.append(torch.FloatTensor([pi]))

            # swap active
            self.log('player: {}, move: {}'.format(turn, final_action.action))
            turn ^= 1
            move += 1

        even_won = 1 if turn == 0 else -1
        for idx in range(move):
            z = even_won if idx % 2 == 0 else -1 * even_won
            zs.append(torch.FloatTensor([z]))

        print(s_node._state.to_string())
        self.log('winner: ', turn + 1)
        mbs = self.send_off(torch.cat(states, 0), torch.stack(pis, 0), torch.stack(zs, 0))
        return mbs

    def forward(self, agent):
        batches = []
        for idx in range(self.num_games):
            disable_resign = True if idx < self.num_games / 10 else False
            batches += self.play_game(agent)
            agent.reset()
        return True

    def run(self):
        while True:
            next_task = self.in_queue.get()
            if next_task is None:
                print('%s: Exiting' % self.name)
                self.in_queue.task_done()
                break
            act, agent, _ = next_task()
            self.log('AT SELF PLAY')
            if act == 'agent':
                agent.num_sims = self.num_sims
                t = self.forward(agent)
            self.in_queue.task_done()
        return



