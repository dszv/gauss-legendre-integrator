import numpy as np

# 8th-order Gauss-Legendre Integrator
def gl8(y, f, dt, eps):
    """
    Perform an 8th-order Gauss-Legendre integration.

    Parameters:
    y  : array-like, the initial values at time t
    f  : function, the function to integrate (e.g., the system of DEs)
    dt : float, the time step for the integration; should be small enough to ensure the Lipschitz condition of function f (see references for a detailed explanation)
    eps: float, tolerance level
    
    Returns:
    array-like : the updated values after the integration step
    """

    # define the weights for the 8th-order Gauss-Legendre method
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
    
    # Butcher-tableau of the Gauss-Legendre method
    A = np.array([[w1m, w1p - w3p + w4m, w1p - w3p - w4m, w1m - w5p],
                  [w1m - w3m + w4p, w1p, w1p - w5m, w1m - w3m - w4p],
                  [w1m + w3m + w4p, w1p + w5m, w1p, w1m + w3m - w4p],
                  [w1m + w5p, w1p + w3p + w4m, w1p + w3p - w4m, w1m]])
    b = np.array([2 * w1m, 2 * w1p, 2 * w1p, 2 * w1m])

    n = len(y)
    g = np.zeros((4, n), dtype=np.float128)

    # iterate to solve for the updated values using the Gauss-Legendre method
    for _ in range(50):
        gprev = g
        g = np.dot(A, g)
        for j in range(4):
            g[j] = f(y + dt * g[j])
        
        # check convergence by comparing the new and previous g values
        error = np.max(np.abs(g - gprev))
        if error < eps:
            break
    
    return y + dt * np.dot(b, g)
