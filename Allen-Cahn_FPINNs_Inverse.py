import random
import time
import os

import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

from plotting import newfig, savefig
from FuzzyLayers import FuzzyLayer

# seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 1111
seed_torch(seed)

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the deep neural network
class DNN(torch.nn.Module):
    # def __init__(self, layers):
    def __init__(self, layers, fuzzy, linear):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layerDict = OrderedDict(layer_list)

        print(layerDict)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        self.fuzzylayer = FuzzyLayer(2, fuzzy)
        self.layer1 = torch.nn.Linear(fuzzy + linear, 1)

    def forward(self, x):
        out1 = self.layers(x)
        out2 = self.fuzzylayer(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.layer1(out)

        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers, fuzzy, linear, lb, ub):
        # def __init__(self, X, u, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # settings
        self.lambda_1 = torch.tensor([1.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)  
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks
        # self.dnn = DNN(layers).to(device)
        self.dnn = DNN(layers, fuzzy, linear).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)  
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),  
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,  
            tolerance_grad=1e-5,  
            tolerance_change=1.0 * np.finfo(float).eps, 
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())  #
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        # print("ushape: " + str(u.shape))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        u = self.net_u(x, t)
        # print("u: " + str(u.shape))

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=torch.ones_like(u_xx),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - lambda_1 * u_xx + lambda_2 * (u ** 3) - lambda_2 * u 
        return f

    def loss_func(self):
        start_time = time.time()

        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)  
        LossList.append(loss)

        self.optimizer.zero_grad()
        loss.backward()  

        self.iter += 1
        if self.iter % 100 == 0:  # 每隔100次打印一次结果
            elapsed = time.time() - start_time

            print(
                'It: %d, Loss: %e, l1: %.5f, l2: %.5f, Time: %.2f' %
                (
                    self.iter,
                    loss.item(),
                    self.lambda_1.item(),
                    self.lambda_2.item(),
                    elapsed
                )
            )
        return loss

    def train(self, nIter):
        self.dnn.train()  
        start_time = time.time()

        for epoch in range(nIter):  
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            LossList.append(loss)

            self.optimizer_Adam.zero_grad()  
            loss.backward()  
            self.optimizer_Adam.step() 

            # 打印训练结果
            if epoch % 100 == 0:
                elapsed = time.time() - start_time

                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        self.lambda_2.item(),
                        elapsed
                    )
                )
                start_time = time.time()

        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

if __name__ == "__main__":

    lambda_1_value = 0.0001
    lambda_2_value = 5

    N_u = 8000  # 初边值点的数量
    data = scipy.io.loadmat('data\\AC.mat')

    t = data['tt'].flatten()[:, None]  # shape:201*1
    print("t: ", t.shape)
    x = data['x'].flatten()[:, None]  # shape:512*1
    print("x: ", x.shape)
    Exact = np.real(data['uu']).T  # shape:201*512
    print("Exact: ", Exact.shape)
    X, T = np.meshgrid(x, t)  # 生成网格采样点矩阵  shape: X:201*512 T:201*512
    print("X: ", X.shape)
    print("T: ", T.shape)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    print("X_star: ", X_star.shape)
    u_star = Exact.flatten()[:, None]
    print("u_star: ", u_star.shape)

    # Doman bounds
    lb = X_star.min(0)  
    ub = X_star.max(0)

    ####################### Training on Non-noisy Data #######################

    # time

    # noise = 0.0

    # # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]  
    print("X_u_train: ", X_u_train.shape)
    u_train = u_star[idx, :]  # label
    print("u_train: ", u_train.shape)

    ################## Training on Noisy Data ####################

    error = []
    # for noise in np.arange(0, 0.06, 0.01):
    #     for fuzzy in np.arange(2, 11, 2):
    #         for linear in np.arange(5, 35, 5):
    noise = 0.0
    fuzzy = 4
    linear = 10
    print("noise:", noise)
    print("fuzzy:", fuzzy)
    print("linear:", linear)
    LossList = []

    # create training set
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    print("u_train: ", u_train.shape)

    # training
    layers = [2, 200, 200, 200, 200, linear]  # 输入2个参数，输出1个参数
    model = PhysicsInformedNN(X_u_train, u_train, layers, fuzzy, linear, lb, ub)
    model.train(10000)

    # evaluations
    u_pred, f_pred = model.predict(X_star)
    # print("u_pred:", type(u_pred))
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
    lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - lambda_1_value) / lambda_1_value * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - lambda_2_value) / lambda_2_value * 100

    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))
    print(lambda_1_value_noisy, lambda_2_value_noisy)

    error.append(str(noise) + '' + str(fuzzy) + ' ' + str(linear) + ' ' + str(error_u) + ' ' + str(
        lambda_1_value_noisy) + ' ' + str(error_lambda_1_noisy) + ' ' + str(lambda_2_value_noisy) + ' ' + str(
        error_lambda_2_noisy))
    root1 = './loss/AC/Inverse/FPINN' + str(noise) + '_' + str(fuzzy) + '_' + str(linear) + '.txt'
    root2 = './predict/AC/Inverse/FPINN' + str(noise) + '_' + str(fuzzy) + '_' + str(linear) + '.csv'
    with open(root1, 'w') as f:
        for j in LossList:
            j = str(j.tolist())
            f.write(j + '\n')

    np.savetxt(root2, u_pred)

    with open('Allen-Cahn_FPINNs_Inverse_Result.txt', 'w') as f:
        for i in error:
            f.write(i + '\n')
            # f.write('\r\n')
