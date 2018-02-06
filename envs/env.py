import argparse
from visdom import Visdom
from aindnn.zero import OptimizeProc, Evaluator
from aindnn.slnn import ZeroNet
from agents.zero_agent import Agent
from envs.isolation.isolation import LTBoard
from multiprocessing import Queue, JoinableQueue
from aindnn.Trainers.trainers import Trainer, MiniBatch, SelfPlay


class IsolationEnvSeq(object):
    def __init__(self, args):
        """
        Enviornment for generic problem

        :param args: (argparge object_
        Attributes
            Processes:
                optim_proc:
                self_play_proc:
                evel_proc:
        """
        self.problem = LTBoard
        self.base_tao = 1
        self.final_tao = 10e-5

        # common - factor out
        self.args =     args                # save a copy of args
        self.verbose =  args.verbose        # to subclasses
        self.vis =      args.vis            # visualizations
        self.epochs =   args.epochs
        self.chkpt_dir = args.chkpt_dir
        self.data_dir = args.data_dir
        self._visdom = None                 # visdom root
        self.start_vis()    # todo seperate vizdom
        # stateful
        self.current_epoch = 0

        # processes
        self.command_control = Queue()
        self.tasks = Queue()
        self.models_to_optim = Queue()
        self.models_to_eval = Queue()

        self.optim_proc = \
            OptimizeProc(self.args,
                         in_queue=self.tasks,
                         out_queue=self.models_to_optim)
        self.self_play_proc = \
            SelfPlay(self.problem,
                     self.args,
                     in_queue=self.models_to_eval,
                     out_queue=self.tasks,
                     num_games=args.num_games_self,
                     num_sims=args.num_sims)
        self.eval_proc = \
            Evaluator(self.problem,
                      self.args,
                      in_queue=self.models_to_optim,
                      out_queue=self.models_to_eval,
                      tao=1e-4,
                      num_sims=args.num_sims,
                      win_threshold=args.win_thresh,
                      num_games=args.num_games_eval)
        # start the processes
        self.optim_proc.daemon = True
        self.self_play_proc.daemon = True
        self.eval_proc.daemon = True
        self.optim_proc.start()
        self.self_play_proc.start()
        self.eval_proc.start()

    def start_vis(self):
        if self.vis is True:
            self._visdom = Visdom()

    def on_new_epoch(self):
        if self.current_epoch == 200:
            mb = MiniBatch('lr', 10e-3, None)
            self.tasks.put(mb)
        if self.current_epoch == 500:
            mb = MiniBatch('lr', 10e-4, None)
            self.tasks.put(mb)
        return

    def forward_sync(self):
        agent = Agent()
        self.reset()
        while self.current_epoch < self.epochs:
            self.on_new_epoch()
            # get best agent
            agent = self.eval_proc.agent
            self.self_play_proc.forward(agent)
            agent.save(iter=self.current_epoch)
            self.current_epoch += 1
        return agent

    def forward(self):
        nn1 = ZeroNet(channels=7, n_blocks=2)
        nn1.save_checkpoint(self.args.best_model)
        nn2 = ZeroNet(channels=7, n_blocks=2)
        nn2.load_checkpoint(self.args.best_model)
        # todo best nn model??????
        a1 = Agent(nn=nn1)
        a2 = Agent(nn=nn2)
        # nn2 = ZeroNet(channels=7, n_blocks=2)
        # self.nn3 = ZeroNet(channels=7, n_blocks=2)
        mb1 = MiniBatch('agent', a1, None)
        mb2 = MiniBatch('agent', a2, None)

        self.tasks.put(mb1)
        self.models_to_optim.put(mb2)

    def reset(self):
        self.optim_proc.reset()
        self.self_play_proc.reset()
        self.eval_proc.reset()

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("echo")
    parser.add_argument('--iters', type=int, default=2)
    parser.add_argument('--q', type=int, default=0, help='[]')
    args = parser.parse_args()
    print(args.echo)

