# Adagrad
import torch

def Adagrad(params, states, hyperparams):
    lr, eps = hyperparams['lr'], hyperparams['eps']
    # perform step()
    for p in params:
        with torch.no_grad():
            # store state to parameter
            param_state = states[p]
            # accumulate past gradients
            if 'accum_grad' not in states[p]:
                param_state['accum_grad'] = torch.square(p.grad)
            else:
                param_state['accum_grad'] += torch.square(p.grad)
            # update parameters
            s = param_state['accum_grad']
            p -= lr * p.grad / torch.sqrt(s + eps)
        # zero gradient
        p.grad.data.zero_()