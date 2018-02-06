import torch
from torch.autograd import Variable
from multiprocessing import Process
import torch.nn as nn
import torch.optim
from aindnn.Trainers.trainers import Trainer, MiniBatch


class OptimizeProc(Trainer, Process):
    """
    Recieves NNs and cached timesteps
        Samples form
    """
    def __init__(self,
                 args,
                 in_queue=None,
                 out_queue=None,
                 agent=None,
                 step=1):
        super(OptimizeProc, self).__init__(args)
        Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.agent = agent

        # visualize
        if self.viz is not None:
            self.init_plots(None)

        # stateful
        self.training = True
        self.n_steps = step
        self.lr = 1e-2

        # losses and optimization
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = None
        if self.agent is not None:
            self.set_optimizer(lr=self.lr)

    def set_optimizer(self, lr=1e-4):
        self.lr = lr
        if self.agent is not None:
            self.optimizer = torch.optim.SGD(
                self.agent.nn.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=10e-4  # with sgd, this = l2 regularization
            )

    def reset(self):
        pass

    def init_plots(self, __na__):
        if self.viz is not None:
            self._plots = \
                dict(l=self.viz.line(
                        X=torch.zeros((1,)).cpu(),
                        Y=torch.zeros((1, 3)).cpu(),
                        opts=dict(xlabel='Iteration', ylabel='Loss',
                                  title='Current Training Loss',
                                  legend=['MSE Loss', 'CrossEnt Loss', 'Loss'])),
                     # acc=self.viz.line(
                     #     X=torch.zeros((1,)).cpu(),
                     #     Y=torch.zeros((1, 1)).cpu(),
                     #     opts=dict(xlabel='Iteration', ylabel='Accuracy',
                     #               title='Optim - Accuracy', legend=['Accuracy']))
                )

    def show_loss(self, k, data):
        """

        :param data:
        :return:
        """
        if self.viz is not None and self.n_steps % self.show_every == 0:
            self.visualize(k, data, self.n_steps)
        return

    def send_off(self):
        self.out_queue.put(MiniBatch('agent', self.agent, None))
        # self.out_queue.join()
        self.agent = None
        return

    def forward(self, data):
        if self.agent is None:
            return True
        s, pi, z = data
        self.optimizer.zero_grad()

        inputs = Variable(s)
        outputs = self.agent.nn(inputs)
        mse, cross_ent = self.loss(outputs, pi, z)
        loss = mse + cross_ent
        loss.backward()

        self.optimizer.step()
        self.n_steps += 1

        if self.viz is not None:
            # todo add accuracy meters for outcome and action
            # todo best topk guesses ??
            p, _ = outputs
            best_idx, _ = p.data.topk(1, dim=1)
            # self.show_loss('acc', [])
            self.show_loss('l', [mse.data[0], cross_ent.data[0], loss.data[0]])

        if self.n_steps % 1000 == 0:
            self.send_off()
        return True

    def loss(self, nn_outputs, pi, z):
        """ loss function and grads
        l = (z - v)^2 - pi^t * log(p)  +  c || theta ||^2
             [ MSE ]   [ Crossentropy ]    [ L2 Loss ]
        """
        p, v = nn_outputs
        z = Variable(z)
        pi = Variable(pi.squeeze().long())

        mse = self.mse(v, z)
        cross_ent = self.cross_entropy(p, pi)

        return mse, cross_ent

    def run(self):
        while True:
            next_task = self.in_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % self.name)
                self.in_queue.task_done()
                break
            s, pi, z = next_task()
            self.log('OPTIM - RECIEVED MESSAGE', self.n_steps)
            if type(s) == str:
                if s == 'lr':
                    self.set_optimizer(lr=pi)
                elif s == 'save':
                    self.agent.save(iter=self.n_steps)
                elif s == 'agent':
                    self.log('OPTIM - agent set')
                    # pi.num_sims = self.num_sims
                    self.agent = pi
                    self.set_optimizer(lr=self.lr)
            else:
                self.forward((s, pi, z))
            self.in_queue.task_done()
        return

