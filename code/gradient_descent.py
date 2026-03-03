import matplotlib.pyplot as plt
import numpy as np

def J(x): 
    return 2*x**2 + x
def daoHam(x):
    return 4*x+1

def gradient_descent(xo=0.1, lr=0.001, iter=100):
    x_history = []
    x_new=xo
    for i in range(iter):
        x_new-=lr*daoHam(x_new)
        x_history.append(x_new)
        if abs(daoHam(x_new))<0.001:
            break
    return x_new, x_history

x_new, x_history =gradient_descent(0.1, 0.001, 500)

def display(x_history):
    J=[]
    for k in x_history:
        y=2*k*k+k
        J.append(y)
    return J

xs = np.linspace(-1, 1, 400)          # pick whatever range you need
plt.plot(xs, J(xs), label='J(x) = 2x²+x')

# optionally overlay the descent trajectory
plt.scatter(x_history, [J(x) for x in x_history],
            color='red', label='gradient‑descent steps')

plt.xlabel('x')
plt.ylabel('J(x)')
plt.legend()
plt.show()