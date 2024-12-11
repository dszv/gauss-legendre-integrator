import numpy as np
import matplotlib.pyplot as plt

# 8th-order gauss-legendre integrator
def gl8(y, f, dt):
    w1p = np.float128(1/8 + np.sqrt(30)/144)
    w1m = np.float128(1/8 - np.sqrt(30)/144)
    w2p = np.float128(1/2 * np.sqrt((15 + 2*np.sqrt(30)) / 35 ))
    w2m = np.float128(1/2 * np.sqrt((15 - 2*np.sqrt(30)) / 35 ))
    w3p = np.float128(( 1/6 + np.sqrt(30)/24 ) * w2p)
    w3m = np.float128(( 1/6 - np.sqrt(30)/24 ) * w2m)
    w4p = np.float128(( 1/21 + 5 * np.sqrt(30)/168 ) * w2p)
    w4m = np.float128(( 1/21 - 5 * np.sqrt(30)/168 ) * w2m)
    w5p = np.float128(w2p - 2 * w3p)
    w5m = np.float128(w2m - 2 * w3m)
    
    A = np.array([[w1m, w1p - w3p + w4m, w1p - w3p - w4m, w1m - w5p],
    [w1m - w3m + w4p, w1p, w1p - w5m, w1m - w3m - w4p],
    [w1m + w3m + w4p, w1p + w5m, w1p, w1m + w3m - w4p],
    [w1m + w5p, w1p + w3p + w4m, w1p + w3p - w4m, w1m]])
    
    b = np.array([2 * w1m, 2 * w1p, 2 * w1p, 2 * w1m])

    n = len(y)
    g = np.zeros((4, n), dtype = np.float128)

    for _ in range(16):
        gprev = g
        g = np.dot(A, g)
        for j in range(4):
            g[j] = f(y + dt * g[j])
        error = np.max(np.abs(g - gprev))
        if error < 1.e-18:
            break
    return y + dt * np.dot(b, g)

# simple pendulum
def f(y):
    theta, omega = y[0], y[1]
    return np.array([omega, -np.sin(theta)])

# initial conditions
y0 = np.array([np.pi/6, 0.1])

# simulation parameters
T = 1000.0
dt = 0.01
TT = int(T / dt)

# state vector initialization
Y = np.zeros((TT, 2), dtype = np.float128)
Y[0] = y0

# time integration
for t in range(1, TT):
    Y[t] = gl8(Y[t - 1], f, dt)

# energy conservation
theta, omega = Y[0, 0], Y[0, 1]
E0 = 0.5 * omega**2 + ( 1 - np.cos(theta) )
dE = np.zeros(TT)

for t in range(1, TT):
    theta, omega = Y[t, 0], Y[t, 1]
    E = 0.5 * omega**2 + ( 1 - np.cos(theta) )
    dE[t] = E - E0

# plots
time = dt*np.arange(TT)
plt.plot(time, dE, color = 'grey')
plt.xlabel(r'$t$')
plt.ylabel(r'$ \Delta E (t) \equiv E(t) - E_0 $')
plt.grid()
plt.savefig("energy_conservation.png")
