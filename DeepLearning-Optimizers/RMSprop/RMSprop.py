# RMSprop
def RMSprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], hyperparams['eps']
    for p in params:
        with torch.no_grad():
            # store state to parameter
            param_state = states[p]
            if 'exp_avg' not in states[p]:
                param_state['exp_avg'] = (1 - gamma) * torch.square(p.grad)
            else:
                param_state['exp_avg'] = gamma * param_state['exp_avg'] + (1 - gamma) * torch.square(p.grad)
            # update p
            s = param_state['exp_avg']
            p -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        # zero gradient
        p.grad.data.zero_()