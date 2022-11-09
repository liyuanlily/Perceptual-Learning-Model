# Reinforcement learning model from Law&Gold, 2009
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import random

def logistic(input, beta):
    """
    Logistic function.
    """
    return 1 / (1 + np.exp(-beta * np.abs(input)))


def stimulus(label=None, theta=None, COH=None, T=None, task='0'):
    """
    Simulate visual stimuli: some dots in a circle, each moving at a direction.

    label: label of the visual stimuli, moving left or right
    theta: the direction of the stimuli moving to
    COH: the percentage of dots moving at the given direction
    T: perception time
    task: task type
    """
    if not theta:
        if task == '0':
            if not label:
                a = np.random.randint(0, 2)
                if a == 0:
                    label = 1 # 1
                    theta = 0
                else:
                    label = -1 # -1
                    theta = np.math.pi
            else:
                theta = 0 if label == 1 else np.math.pi
        else:
            if not label:
                a = np.random.randint(0, 2)
                if a == 0:
                    label = 1 # 1
                    theta = 10 / 180 * np.math.pi
                else:
                    label = -1 # -1
                    theta = -10 / 180 * np.math.pi
            else:
                theta = 10 / 180 * np.math.pi if label == 1 else -10 / 180 * np.math.pi
    else:
        theta = theta
        label = None

    if not COH:
        COH = np.random.random()
    elif type(COH) == list:
        COH = np.random.uniform(COH[0], COH[1])

    if not T:
        T = np.random.exponential(scale=0.8)
        if T < 0.1:
            T = 0.1
        elif T > 1.4:
            T = 1.4
    return [theta, COH, T], label


def gauss(x, mu=0, sigma=1):
    """
    Compute gaussian distribution.
    """
    res = 1 / np.sqrt(2*np.math.pi) / sigma * np.exp(-(x - mu)**2 / 2 / sigma**2)
    return res


class MT_Neurons():
    """
    A model for direction-selective sensory neurons in the middle temporal cortex (of monkey cortex).
    """
    def __init__(self, group_num=36, units_num=7200, k_n=0.5, k_0=1.0,
                 k_p=6.0, rho=0.5, b_sen=90/180*np.math.pi, b_dir=30/180*np.math.pi):
        """
        Initiating parameters.

        group_num: number of groups according to different direction tuning functions
        units_num: number of all neurons
        """
        self.group_num = group_num
        self.group_units = int(units_num / group_num)
        self.k_n = k_n
        self.k_0 = k_0
        self.k_p = k_p
        # sigma_unit = np.linspace(math.log(np.math.pi), math.log(1.1), 200)
        # sigma_unit = np.exp(np.sqrt(sigma_unit**3))
        sigma_unit = np.sqrt(np.linspace((60/180*np.math.pi)**2, (1/180*np.math.pi)**2, 200))
        # sigma_unit = np.linspace(np.sqrt(90/180*np.math.pi), np.sqrt(1/180*np.math.pi), 200)
        # sigma_unit = np.linspace(np.math.pi, 0, 200)
        self.sigma = np.zeros(units_num)
        for i in range(self.group_num):
            self.sigma[i*self.group_units:(i+1)*self.group_units] = sigma_unit
        self.centers = np.zeros(units_num)
        self.units_num = units_num
        group_units = int(units_num / group_num)
        # sen = 1 / np.exp(np.linspace(math.log(0.8), math.log(0.06), 200)) # 0.06 0.8
        # self.sen = np.zeros(units_num) # the inverse of COH
        # assert type(self.sen) == np.ndarray
        for i in range(group_num):
            self.centers[i * group_units : (i+1) * group_units] = (-170 + i * 10) / 180 * np.math.pi
            # self.sen[i * group_units : (i+1) * group_units] = sen[:]
        # correlation matrix
        self.G_sen = rho - np.abs(self.sigma.reshape((units_num, 1)) - self.sigma.reshape((1, units_num))) / b_sen
        assert self.G_sen.shape == (units_num, units_num)
        self.G_sen[self.G_sen<=0] = 0
        self.G_dir = np.exp(-np.abs(self.centers.reshape((units_num, 1)) - self.centers.reshape((1, units_num))) / b_dir)
        assert self.G_dir.shape == (units_num, units_num)
        self.R = self.G_sen * self.G_dir
        # self.R = self.G_dir
        for i in range(units_num):
            self.R[i, i] = 1
        # self.U = sp.linalg.cholesky(self.R)
        self.x = None
        self.m = None
        self.v = None

    def forward(self, stimulus):
        """
        Compute the respond of each MT-neuron to be sent to next layer.

        stimulus: visual stimulus
        """
        theta, COH, T = stimulus

        # old x:
        self.m = np.exp(-(theta - self.centers) ** 2 / (2 * self.sigma ** 2))
        m = T * (self.k_0 + COH * (self.k_n + (self.k_p - self.k_n) * self.m))
        v = 2 * m
        self.v = v
        z = np.random.standard_normal(size=self.units_num)
        # z = np.zeros(self.units_num)
        # z = np.ones(self.units_num)
        r = (self.R @ z.reshape((z.shape[0], 1))).reshape(z.shape[0])
        x = m + np.sign(r) * np.sqrt(v * np.abs(r))
        Ex = T * self.k_0 + np.sign(r) * np.sqrt(2 * m * np.abs(r))

        # # modified x (my version):
        # self.m = np.exp(-(theta - self.centers) ** 2 / (2 * self.sigma ** 2))
        # m = T * (self.k_0 + COH * (self.k_n + (self.k_p - self.k_n) * self.m))
        # x = random.gauss(mu=m, sigma=np.sqrt(v))
        # Ex = m

        # # new x:
        # self.m = np.exp(-(theta - self.centers) ** 2 / (2 * self.sigma ** 2))
        # m = T * (self.k_0 + COH * (self.k_n + (self.k_p - self.k_n) * self.m))
        # x = m
        # Ex = None
        self.x = x

        return x, Ex


# one pool of neurons   alternative: two pools of neurons, comparable results to the one poll
class Pooling():
    """
    Pooling layer.
    """
    def __init__(self, learning_rate, m=1, n=1, wamp=0.02, learning_rule='rein', pooling_type='linear'):
        """
        Initiating parameters.
        """

        self.w = np.random.uniform(-1, 1, 7200)
        self.w = self.w / (np.linalg.norm(self.w) / wamp)

        plt.imshow(self.w.reshape(36, 200).T, extent=[-170, 180, 0, 200], aspect='auto')
        plt.colorbar()
        plt.xlabel('Direction tuning(deg)')
        plt.ylabel('Threshold(% coh)')
        plt.title('pooling weight at beginning')
        plt.savefig('W0.jpg')
        plt.show()

        self.learning_rate = learning_rate
        self.n = n
        self.m = m
        self.wamp = wamp
        self.beta = 0.1
        self.learning_rule = learning_rule
        self.pooling_type = pooling_type

    def forward(self, x):
        """
        Compute the sensory output.

        x: response of each neurons
        output: sensory output, computed as the sum of response of all neurons multipling their corresponding weights
        """
        if self.pooling_type == 'linear':
            # linear pooling
            y = np.sum(x * self.w)
        else:
            # nonlinear pooling
            # 1.19
            y = np.sum(x**1.19 * self.w)
        # # 1.41
        # y = np.sum(x**1.41 * self.w)
        # # 2
        # y = np.sum(x**2 * self.w)

        # additive noise & multiplicative noise
        y = y + np.random.normal(loc=0, scale=5, size=1)
        y = y + np.random.normal(loc=0, scale=float(np.sqrt(2*np.abs(y))), size=1)
        # other noise type
        # # only additive
        # y = y + np.random.normal(loc=0, scale=5, size=1)
        # # only multiplicative
        # y = y + np.random.normal(loc=0, scale=float(np.sqrt(2*np.abs(y))), size=1)

        return float(y)

    def backward(self, x, y, label, choice, Ex):
        """
        Compute the loss between prediction and actual label, then adjust the weights according to the loss.

        label: label of the visual stimulus, -1 or 1
        choice: prediction, -1 or 1

        return:
        error: loss between prediction and actual label
        Ey: the estimated probability of a correct decision
        """
        Ey = logistic(y, self.beta)
        error = (1 if label == choice else 0) - self.m * Ey
        assert type(label) == int and type(choice) == int

        if self.learning_rule == 'rein':
            # reinforcement learning rule
            if (label==-1 and choice==-1) or (label==1 and choice==1):
                dw = self.learning_rate * label * error * (x - self.n * np.average(x))
            else:
                dw = self.learning_rate * label * error * (self.n * np.average(x) - x)
        else:
            # Oja-like learning rule
            if (label == -1 and choice == -1) or (label == 1 and choice == 1):
                dw = self.learning_rate * label * (error * (x - self.n * np.average(x)) - error**2*self.w/math.sqrt(0.02))
            else:
                dw = self.learning_rate * label * error * ((self.n * np.average(x) - x) - error**2*self.w/math.sqrt(0.02))

        # with normalization and is constant
        self.w = self.w + dw
        self.w = self.w / (np.linalg.norm(self.w) / math.sqrt(self.wamp))
        # # no normalization
        # self.w = self.w + dw
        # # substractive normalization
        # self.w = self.w + dw
        # self.w = self.w / np.sum(self.w) * self.wamp

        return error, Ey


def Decision(y):
    """
    Decision function. If y>0, then choose 1; otherwise, choose -1.
    """
    if y > 0:
        return 1
    else:
        return -1


def thres_lapse(neurons, pooling, task):
    """
    Compute threshold(using 1000 trial blocks) when the stimulus can just be correctly perceived,
    and lapse rate(using 250 trial blocks).
    """
    correct_pers = []
    threshold = None
    for i in range(1, 100):
        stimuli, label = stimulus(COH=i/100, T=1, task=task)
        correct = 0
        for t in range(100):
            x, Ex = neurons.forward(stimuli)
            y = 0
            for j in range(10):
                y += pooling.forward(x)
            y /= 10
            choice = Decision(y)
            correct += 1 if label==choice else 0
        correct_pers.append(correct/100)
        if correct_pers[-1] > 0.81:
            threshold = i
            break
    if not threshold:
        threshold = 99.9

    # compute lapse rate
    lapse_rate = 0
    for i in range(250):
        stimuli, label = stimulus(COH=0.999, task=task)
        x = neurons.forward(stimuli)[0]
        y = 0
        for t in range(10):
            y += pooling.forward(x)
        y /= 10
        choice = Decision(y)
        Ey = logistic(y, pooling.beta)
        lapse_rate += (1 if label == choice else 0) - pooling.m * Ey
    lapse_rate /= 250
    return threshold, lapse_rate


def w_vs_time(neurons, pooling, epoch, task, name):
    """
    Plot how response change with time.
    """
    xs = np.linspace(0, 1, 10)
    for COH in [99.9, 51.2, 25.6, 12.8, 6.4, 3.2, 0.0]:
        ys = []
        for t in np.linspace(0, 1, 10):
            stimuli, label = stimulus(COH=COH, T=t, label=1, task=task)
            y = 0
            for i in range(100):
                x = neurons.forward(stimuli)[0]
                y += pooling.forward(x)
            y /= 100
            ys.append(y)
        plt.plot(xs, ys, label=str(COH)+'% coh')
        ys = []
        for t in np.linspace(0, 1, 10):
            stimuli, label = stimulus(COH=COH, T=t, label=-1, task=task)
            y = 0
            for i in range(100):
                x = neurons.forward(stimuli)[0]
                y += pooling.forward(x)
            y /= 100
            ys.append(y)
        plt.plot(xs, ys, label=str(COH) + '% coh', linestyle='--')
    plt.xlabel('Time(s)')
    plt.ylabel('Response(spikes per s)')
    plt.ylim(-100, 100)
    plt.legend()
    plt.savefig(name+'wtime'+str(epoch)+'.jpg')
    plt.show()


class Queue():
    """
    Queue, a kind of data structure.
    """
    def __init__(self, size):
        """
        size: maximum size of the queue
        """
        self.queue = []
        self.size = size

    def add(self, x):
        """
        Add an element into the queue.
        """
        if len(self.queue) == self.size:
            del self.queue[0]
        self.queue.append(x)


def im_show(x):
    """
    Use plt.imshow() to show the response of MT-neurons.
    """
    plt.imshow(x.reshape(36,200).T, extent=[-170, 180, 0, 200], aspect='auto')
    plt.colorbar()
    plt.show()


def optimal_pooling(neurons, task, name):
    """
    Compute and show optimal pooling weight.
    """
    COH = np.average(np.exp(np.linspace(math.log(0.8), math.log(0.06), 200)))
    stimuli = stimulus(label=1, COH=COH, T=1, task=task)[0]
    neurons.forward(stimuli)
    m_R = neurons.m
    v_R = neurons.v
    stimuli = stimulus(label=-1, COH=COH, T=1, task=task)[0]
    neurons.forward(stimuli)
    m_L = neurons.m
    v_L = neurons.v
    m = m_R - m_L
    Sigma = np.zeros((7200,7200))
    for i in range(7200):
        Sigma[i, i] = v_R[i] + v_L[i]
    w_opt = (np.linalg.inv(Sigma) @ m.reshape((7200, 1))).reshape(7200)
    w_opt = w_opt / math.sqrt(np.sum(w_opt**2) / 0.02)
    plt.imshow(w_opt.reshape(36, 200).T, extent=[-170, 180, 0, 200], aspect='auto')
    plt.colorbar()
    plt.title('Optimal pooling weight')
    plt.savefig(name+'OPtW.jpg')
    plt.show()


def training(task, learning_rule, pooling_type):
    """
    Training process.
    """
    name = task + learning_rule + pooling_type
    # optimal_pooling(neurons, task, name)
    pooling = Pooling(10**(-5), learning_rule=learning_rule, pooling_type=pooling_type)
    print('pooling ready')
    epoch_num = 100
    accs = np.zeros(100)
    es = np.zeros(100)
    cprs = np.zeros(100)
    thresholds = np.zeros(epoch_num)
    lapse_rates = np.zeros(epoch_num)
    inputs = Queue(300)
    rewards = Queue(300)
    for epoch in range(100):
        print('epoch', epoch)
        errors = []
        acs = []
        er = 0
        acc = 0
        cpr = []
        cps = 0
        beta = 1
        for turn in range(1000):
            if 0 <= epoch < 40:
                stimuli, label = stimulus(COH=[0.9, 1], task=task)
            elif 40 <= epoch <80:
                stimuli, label = stimulus(COH=[0.5, 1], task=task)
            else:
                stimuli, label = stimulus(task=task)

            x, Ex = neurons.forward(stimuli)
            y = pooling.forward(x)
            choice = Decision(y)
            reward = 1 if choice == label else 0
            inputs.add(y)
            if reward == 0:
                rewards.add(0.5)
            else:
                rewards.add(0.55)
            error, cp = pooling.backward(x, y, label, choice, Ex)
            acc += 1 if label == choice else 0
            er += float(error)
            cps += cp
            if turn % 300 == 299:
                input = np.array(inputs.queue)
                output = np.array(rewards.queue)
                popt, pcov = curve_fit(logistic, input, output)
                print(popt)
                pooling.beta = float(popt)
            # beta = curve_fit(logistic, [y], [reward])[0] # update beta trial by trial
            # how to update trial by trial?

            if turn % 20 == 19:
                errors.append(er / 20)
                acs.append(acc / 20)
                cpr.append(cps / 20)
                acc = 0
                er = 0
                cps = 0
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(36, 200).T / np.sqrt(np.sum(x**2)), extent=[-170, 180, 0, 200], aspect='auto')
        # plt.ylim(0, 40)
        plt.title('MT population responses')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(pooling.w.reshape(36, 200).T, extent=[-170, 180, 0, 200], aspect='auto')
        # plt.ylim(0, 40)
        plt.title('Pooling weights')
        plt.colorbar()
        plt.savefig(name+'RW' + str(epoch) + '.jpg')
        plt.show()

        plt.subplot(3, 1, 1)
        plt.plot(errors)
        plt.ylabel('errors')
        plt.subplot(3, 1, 2)
        plt.plot(acs)
        plt.ylabel('accuracy')
        plt.ylim(0, 1)
        plt.subplot(3, 1, 3)
        plt.plot(cpr)
        plt.xlabel('Turns')
        plt.ylabel('choice probability')
        plt.ylim(0, 1)
        plt.suptitle('Epoch'+str(epoch))
        plt.savefig(name+'epoch'+str(epoch)+'.jpg')
        plt.show()
        print(epoch, 'accuracy', np.mean(acs), 'error', np.mean(errors))
        accs[epoch] = np.mean(acs)
        es[epoch] = np.mean(errors)
        cprs[epoch] = np.mean(cpr)

        # Plot threshold：the motion coherence corresponding to 81% correct responses at a viewing duration of 1 s
        threshold, lapse_rate = thres_lapse(neurons, pooling, task)
        thresholds[epoch] = threshold
        lapse_rates[epoch] = lapse_rate

        # Pot pooling weight
        # fig3a in the paper
        plt.imshow(pooling.w.reshape(36, 200).T, extent=[-170, 180, 0, 200], aspect='auto')
        plt.xlabel('Direction tuning(deg)')
        plt.ylabel('Threshold(% coh)')
        plt.title('pooling weight at epoch'+str(epoch))
        plt.colorbar()
        # plt.savefig('w_epoch'+str(epoch)+'.jpg')
        plt.savefig(name+'W' + str(epoch) + '.jpg')
        plt.show()
        # fig3c in the paper
        # the 30th
        plt.subplot(2, 1, 1)
        z = pooling.w.reshape(36, 200)[:, 29]
        x = np.linspace(-170, 180, 36)
        plt.scatter(x, z.reshape(36))
        plt.ylabel('Pooling weight')
        plt.ylim(-0.005, 0.005)
        plt.title('8.75% coh')
        # the 84th
        plt.subplot(2, 1, 2)
        z = pooling.w.reshape(36, 200)[:, 150]
        plt.scatter(x, z.reshape(36))
        plt.ylabel('Pooling weight')
        plt.ylim(-0.005, 0.005)
        plt.xlabel('Direction tuning(deg)')
        plt.title('41.8% coh')
        plt.savefig(name+'RWCoh' + str(epoch) + '.jpg')
        plt.show()
        # Fig5a in the paper: Mean pooled responses as a function of viewing time
        w_vs_time(neurons, pooling, epoch, task, name)

        # supplementary fig4c in the paper
        # plot the mean response (supplementary fig4c in the paper)
        # average over 100 times
        x_999 = np.zeros(7200)
        x_256 = np.zeros(7200)
        x_064 = np.zeros(7200)
        for k in range(100):
            stimuli = stimulus(COH=0.999, task=task)[0]
            x_999 += neurons.forward(stimuli)[0]
        x_999 = x_999 / 100
        plt.plot(np.linspace(-170, 180, 36), np.sum(x_999.reshape(36, 200), axis=1), label='COH99.9%')
        for k in range(100):
            stimuli = stimulus(COH=0.256, task=task)[0]
            x_256 += neurons.forward(stimuli)[0]
        x_256 = x_256 / 100
        plt.plot(np.linspace(-170, 180, 36), np.sum(x_256.reshape(36, 200), axis=1), label='COH25.6%')
        for k in range(100):
            stimuli = stimulus(COH=0.064, task=task)[0]
            x_064 += neurons.forward(stimuli)[0]
        x_064 = x_064 / 100
        plt.plot(np.linspace(-170, 180, 36), np.sum(x_064.reshape(36, 200), axis=1), label='COH6.4%')
        plt.xlabel('Motion direction(deg)')
        plt.ylabel('Response(spikes per s)')
        plt.legend()
        plt.savefig(name+'Rs' + str(epoch) + '.jpg')
        plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(es)
    plt.ylabel('error')
    plt.subplot(2, 1, 2)
    plt.plot(accs)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.savefig(name+'All.jpg')
    plt.show()

    # fig2b in the paper
    plt.subplot(2, 1, 1)
    plt.plot(thresholds)
    plt.xlabel('Trial(*1,000)')
    plt.ylabel('Threshold(% COH)')
    plt.subplot(2, 1, 2)
    plt.plot(lapse_rates)
    plt.xlabel('Trial(*1,000)')
    plt.ylabel('Lapse rate')
    plt.savefig(name+'thres_lapse.jpg')
    plt.show()


if __name__ == '__main__':
    # train
    # first high coherence stimuli, later introduce low-coherence stimuli
    neurons = MT_Neurons(36, 7200)
    print('neurons ready')
    # testing
    stimuli=[0,1,1.4]
    x=neurons.forward(stimuli)[0]
    im_show(x)

    training(task='0', learning_rule='rein', pooling_type='linear')
    # training(task='0', learning_rule='oja', pooling_type='linear')
    # training(task='0', learning_rule='rein', pooling_type='nonlinear')