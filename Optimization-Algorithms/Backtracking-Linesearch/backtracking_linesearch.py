import numpy as np

# backtrack for Newton's method

# func is objective function, x_k is input matrix, g_func is derivative of objective function
# h_func is hessian of objective function. c, t and pho are hyperparameters for backtracking
# line search
def backtrack(func, x_k, g_func, h_func, c, t, pho):
    d_k = -np.linalg.inv(h_func(x_k))@g_func(x_k)
    while func(x_k + t*d_k) - func(x_k) >= c*t*(g_func(x_k).T@d_k):
        t *= pho
    return t

# gradient descent

# bt is a boolean to determine if backtracking should continue, obj_vals track the output
# of objective function
def gradient_descend(func, g_func, h_func, x_k, bt, obj_vals, t, c, pho):
    # d_k = (hessian)^-1(gradient)
    d_k = -np.linalg.inv(h_func(x_k))@g_func(x_k)
    if bt == True:
        t_bt = backtrack(func, x_k, g_func, h_func, c, t, pho)
        print(f"backtracking learning rate: {t_bt}")
        x_new = x_k + t_bt*d_k
    else:
        x_new = x_k + t*d_k
    print(f"objective value: {func(x_k)}")
    obj_vals.append(func(x_k))
    # stopping criterion: check gradient norm < 1e-5 to stop
    if func(x_k) < 1e-5:
        return x_new
    else:
        return gradient_descend(func, g_func, h_func, x_new, bt, obj_vals, t, c, pho)