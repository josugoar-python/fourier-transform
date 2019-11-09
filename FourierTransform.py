# DISCRETE FOURIER TRANSFORM

import numpy as np
import matplotlib.pyplot as plt

global N
N = 6000

# X(k): transformed sample value (complex domain)
# x(n): input data sample value (real or complex domain)
# N:    number of samples
# M:    N-th primitive root of unity

# Discrete Fourier Transform
def DFT(x):
    # Arrange x(n) n stacks
    n = np.arange(N)
    # Arrange X(k) k stacks
    k = n.reshape((N, 1))
    # Calculate M
    M = np.exp(-2j * np.pi * n * k / N)
    # Calculate X(k) stack values
    X = np.dot(M, x)
    return X

# t: function linspace
# T: sampling period
# f: sampling frequency

def func(limits):
    t = np.linspace(limits[0], limits[1], N)
    T = t[1] - t[0]
    f = np.linspace(0, 1 / T, N)
    return (np.cos(2*np.pi*(3*t)))*(np.exp(-np.pi*t**2)), t, f
    # return np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t), t, f

x, t, f = func(limits=[-2, 2])
X = DFT(x=x)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)
ax1.plot(t, x)
ax2.bar(f[: N // 240], np.abs(X)[: N // 240], width=0.05)

plt.show()
