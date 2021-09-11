import numpy as np

# Gauss_Newton method with recursive implementation

# func is objective function, g_func is Jacobian of objective function
# x_k is the input matrix, obj_vals is the tracking for objective values
def Gauss_Newton(func, g_func, x_k, obj_vals):
    # d_k = (hessian)^-1(gradient)
    # calculate Jacobian
    Jr = np.zeros((m,n))
    for idx in range(m):
        Jr[idx] = grad_gn(x_k, idx)
    # -(Jr.T@Jr)^-1@Jr.T@f(x)
    d_k = -np.linalg.inv(Jr.T@Jr)@Jr.T@f_gn(x_k)
    x_new = x_k + d_k
    print(f"objective value: {func(x_new)}")
    obj_vals.append(func(x_new))
    # stopping criterion: check gradient norm < 1e-5 to stop
    if func(x_new) < 1e-5:
        return x_new
    else:
        return Gauss_Newton(func, g_func, x_new, obj_vals)