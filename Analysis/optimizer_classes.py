import torch

from losses import *

from optimizers.optim import SGD, Adam, SGD_C, Adam_C, RMSprop, RMSprop_C
from optimizers.optimExperimental import SAGA
from torch.autograd import Variable


class Optimizer:
    def __init__(self, opt_name = 'Adam', func_name = 'Beale', lr = 1e-4):
        if func_name == 'Beale':
            self.w1 = torch.nn.Parameter(torch.FloatTensor([0.75]))
            self.w2 = torch.nn.Parameter(torch.FloatTensor([1.0]))
            lr = 1e-3
        elif func_name == 'GoldsteinPrice':
            self.w1 = torch.nn.Parameter(torch.FloatTensor([-1.0]))
            self.w2 = torch.nn.Parameter(torch.FloatTensor([0.0]))
            lr = 1e-5
        elif func_name == 'SixHumpCamel':
            self.w1 = torch.nn.Parameter(torch.FloatTensor([3.0]))
            self.w2 = torch.nn.Parameter(torch.FloatTensor([2.5])) #(3,2)
            lr = 1e-4
        self.name = opt_name

        W = [self.w1, self.w2]

        topC = 5
        decay = 0.99

        
        if self.name == 'SGD':
            self.opt = SGD(W,lr = lr)
            self.color = "y"
        elif self.name == 'SGDM':
            self.opt = SGD(W,lr = lr, momentum = 0.9)
            self.color = "b"
        elif self.name == 'SGD_C':
            self.opt = SGD_C(W,lr = lr, decay=decay, topC = topC)
            self.color = "r"
        elif self.name == 'SGDM_C':
            self.opt = SGD_C(W,lr = lr, momentum = 0.9, decay=decay, topC = topC)
            self.color = "g"
        elif self.name == 'Adam_C':
            self.opt = Adam_C(W, lr = lr, decay=decay, topC = topC)
            self.color = "k"
        elif self.name == 'Adam':
            self.opt = Adam(W, lr = lr)
            self.color = "c"
        elif self.name == 'RMSprop':
            self.opt = RMSprop(W, lr = lr)
            self.color = "bisque"
        elif self.name == 'RMSprop_C':
            self.opt = RMSprop_C(W, lr = lr, decay=decay, topC = topC)
            self.color = "m"
        self.w1_list = []
        self.w2_list = []
        self.loss_list = []
        self.loss_func = eval(func_name)() # 2 variable functions

    def train_step(self):
        self.update()

        self.opt.zero_grad()
        loss = self.loss_func.get_func_val([self.w1, self.w2])
        loss.backward()
        self.opt.step()

        return loss

    def update(self):
        self.w1_list += [float(self.w1)]
        self.w2_list += [float(self.w2)]
        self.loss_list += [float(self.loss_func.get_func_val([self.w1, self.w2]))]

    def plot(self, ax):
        #ax.quiver(self.w1_list[:-1], self.w2_list[:-1], [ a-b for a,b in zip(self.w1_list[1:],self.w1_list[:-1])], \
         #[ a-b for a,b in zip(self.w2_list[1:],self.w2_list[:-1])],\
          #scale_units='xy', angles='xy', scale=1, color=self.color)
         ln, = ax.plot(self.w1_list[:-1], self.w2_list[:-1], self.color+'->', lw = 1, label = self.name)
         return ln

        # ax.plot(self.w1_list,
        #         self.w2_list,
        #         self.loss_list,
        #         linewidth=0.5,
        #         label=self.name,
        #         color=self.color)
        # ax.scatter(self.w1_list[-1],
        #            self.w2_list[-1],
        #            self.loss_list[-1],
        #            s=3, depthshade=True,
        #            label=self.name,
        #            color=self.color)
