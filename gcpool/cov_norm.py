

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np


def symmetric(A):
    return 0.5 * (A + A.t())

def orthogonal_projection(dLdW, Wt):
    out = dLdW - Wt.mm(symmetric(Wt.transpose(0, 1).mm(dLdW)))
    return out

def retraction(dLdW, Wt):
    Wt1 = dLdW + Wt
    Q, R = Wt1.qr()
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out

def compute_grad_input(g, s, u, var_dldu, var_dlds):
    g = symmetric(g)
    eye = g.new(g.size(1))
    eye.fill_(1)
    eye = eye.diag()

    dLdU = 2 * (g.mm(u.mm(var_dldu)))
    dLdS = eye * (var_dlds.mm(u.t().mm(g.mm(u))))

    P = s.unsqueeze(1)
    P = P.expand(-1, P.size(0))
    P = P - P.t()
    mask_zero = torch.abs(P) == 0
    P = 1 / P
    P[mask_zero] = 0
    grad_input = u.mm(symmetric(P.t() * u.t().mm(dLdU)) + dLdS).mm(u.t())

    return grad_input

class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size - input_size), dtype=np.float32)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)
        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))

        return output

# The parameter of BiMap layer
class StiefelParameter(nn.Parameter):

    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class StiefelMetaOptimizer(object):

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is None:
                continue
            if isinstance(p, StiefelParameter):
                if id(p) not in self.state:
                    self.state[id(p)] = p.data.clone()
                else:
                    self.state[id(p)].fill_(0).add_(p.data)
                trans = orthogonal_projection(p.grad.data, p.data)
                p.grad.data.fill_(0).add_(trans)

        loss = self.optimizer.step(closure)
        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is None:
                continue
            if isinstance(p, StiefelParameter):
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                trans = retraction(-lr * p.grad.data, self.state[id(p)])
                p.data.fill_(0).add_(trans)
        return loss


# Bilinear Map to transform the input SPD matrices to new SPD matrices
class BiMapSPDMatrix(nn.Module):

    def __init__(self, input_size, output_size):
        super(BiMapSPDMatrix, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))
        return output


# Logarithm SPD Matrix Function
class LOGSPDMatrixFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.epsilon = epsilon
        output = input.new(input.size(0), input.size(1), input.size(2))
        u_matrix = input.new(input.size(0), input.size(1), input.size(2))
        s_matrix = input.new(input.size(0), input.size(1))
        s_matrix_reEig = input.new(input.size(0), input.size(1))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            u_matrix[k] = u
            s_matrix[k] = s
            s[s < epsilon] = epsilon
            s_matrix_reEig[k] = s
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))
        ctx.save_for_backward(input, u_matrix, s_matrix, s_matrix_reEig)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input, u_matrix, s_matrix, s_matrix_reEig = ctx.saved_variables
        epsilon = ctx.epsilon
        grad_input = input.new(input.size(0), input.size(1), input.size(2))
        for k, g in enumerate(grad_outputs):
            u = u_matrix[k]
            s = s_matrix_reEig[k]
            s_log = s.log().diag()
            s_inv = (1 / s).diag()
            grad_reEig = compute_grad_input(g, s, u, s_log, s_inv)

            s = s_matrix[k]
            max_mask = s > epsilon
            s_max_diag = s.clone()
            s_max_diag[~max_mask] = epsilon
            s_max_diag = s_max_diag.diag()
            Q = max_mask.float().diag()
            grad_input[k] = compute_grad_input(grad_reEig, s, u, s_max_diag, Q)
        return grad_input, None

class LOGSPDMatrix(nn.Module):
    def __init__(self, epsilon = 1e-4):
        super(LOGSPDMatrix, self).__init__()
        self.epsilon = epsilon

    def forward(self, input):
        output = LOGSPDMatrixFunction.apply(input, self.epsilon)
        return output


# Matrix Power Normalized SPD Matrix Function
class MPNSPDMatrixFunction(Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        output = input.new(input.size(0), input.size(1), input.size(2))
        u_matrix = input.new(input.size(0), input.size(1), input.size(2))
        s_matrix = input.new(input.size(0), input.size(1))
        s_matrix_reEig = input.new(input.size(0), input.size(1))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            u_matrix[k] = u
            s_matrix[k] = s
            s = s**alpha
            s_matrix_reEig[k] = s
            output[k] = u.mm(s.diag().mm(u.t()))
        ctx.save_for_backward(input, u_matrix, s_matrix, s_matrix_reEig)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input, u_matrix, s_matrix, s_matrix_reEig = ctx.saved_variables
        alpha = ctx.alpha
        grad_input = input.new(input.size(0), input.size(1), input.size(2))
        for k, g in enumerate(grad_outputs):
            u = u_matrix[k]
            s = s_matrix[k]
            s_reEig = s_matrix_reEig[k].diag()
            Q = alpha*(s**(alpha-1)).diag()
            grad_input[k] = compute_grad_input(g, s, u, s_reEig, Q)
        return grad_input, None

class MPNSPDMatrix(nn.Module):
    def __init__(self, alpha = 0.5):
        super(MPNSPDMatrix, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        output = MPNSPDMatrixFunction.apply(input, self.alpha)
        return output


# Fast Matrix Power Normalized SPD Matrix Function
class FastMPNSPDMatrixFunction(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None

class FastMPNSPDMatrix(nn.Module):
    def __init__(self, iterN = 5):
        super(FastMPNSPDMatrix, self).__init__()
        self.iterN = iterN

    def forward(self, input):
        output = FastMPNSPDMatrixFunction.apply(input, self.iterN)
        return output
