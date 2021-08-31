# Adam
def Adam(params, states, hyperparams):
    beta1, beta2 = hyperparams['betas']
    lr, eps = hyperparams['lr'], hyperparams['eps']
    for p in params:
        with torch.no_grad():
            # store state to parameter
            param_state = states[p]
            if ('moment_1' and 'moment_2' and 'step') not in states[p]:
                param_state['moment_1'] = (1 - beta1) * torch.square(p.grad)
                param_state['moment_2'] = (1 - beta2) * torch.square(p.grad)
                param_state['step'] = 1
            else:
                param_state['moment_1'] = beta1 * param_state['moment_1'] + (1 - beta1) * p.grad
                param_state['moment_2'] = beta2 * param_state['moment_2'] + (1 - beta2) * torch.square(p.grad)
                param_state['step'] += 1
            m = param_state['moment_1']
            v = param_state['moment_2']
            t = param_state['step']
            m_bias_corr = m / (1 - beta1 ** t)
            v_bias_corr = v / (1 - beta2 ** t)
            # update p
            p -= hyperparams['lr'] * m_bias_corr / (torch.sqrt(v_bias_corr) + eps)
        # zero gradient
        p.grad.data.zero_()