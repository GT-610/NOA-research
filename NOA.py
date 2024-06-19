import random
import numpy as np
from levy import levy

def initialize_population(N, D, lb, ub):
    """
    Initialize the population positions.
    Args:
        N (int): Number of individuals in the population.
        D (int): Number of dimensions in the search space.
        lb (numpy.ndarray): Lower bounds for each dimension (Lj).
        ub (numpy.ndarray): Upper bounds for each dimension (Uj).
        
    Returns:
        initial_positions (numpy.ndarray): Initial positions of the population in the search space.
    """
    rm = np.random.rand(N, D)
    initial_positions = lb + (ub - lb) * rm
    return initial_positions

def nutcracker_optimizer(N, D, lb, ub, T, delta, Prp, fobj):
    """
    Full Nutcracker Optimization Algorithm (NOA) process integrating both foraging-storage and cache-search-recovery strategies.
    Args:
        N, D, lb, ub (int, int, numpy.ndarray, numpy.ndarray): Same as previous definitions.
        T (int): Maximum number of generations for the entire process.
        delta (float): Threshold for exploration decision in the second strategy.
        Prp (float) : Probability employed to determine the percentage of globally exploring other regions within the search space.
        fobj (function): Objective function.
    Returns:
        xbest_final (numpy.ndarray): The best position found.
        best_fit (numpy.float64): The solution / best fitness.
        best_fit_per_gen (numpy.ndarray): Best fitness for each generation.
    """
    positions = initialize_population(N, D, lb, ub)
    xbest = np.zeros(D)
    best_fit = np.inf
    lfit = np.inf * np.ones(N)
    lbest = positions.copy()
    RP = np.zeros((2, D))
    fit = np.inf * np.ones(N)

    for i in range(N):
        fit[i] = fobj(positions[i])
        lfit[i] = fit[i]
        if fit[i] < best_fit:
            best_fit = fit[i]
            xbest = positions[i].copy()

    best_fit_per_gen = np.zeros(T)

    for t in range(1, T+1):
        RL = 0.05 * levy(N, D, 1.5)
        l = np.random.rand() * (1 - t / T)
        alpha = (t / T) ** (2 / t) if np.random.rand() < np.random.rand() else (1 - t / T) ** (2 * t / T)
        
        if np.random.rand() < np.random.rand():
            for i in range(N):
                mu = np.random.rand() if np.random.rand() < np.random.rand() else np.random.randn() if np.random.rand() < np.random.rand() else RL[0][0]
                A, B, C = np.random.choice(N, 3, replace=False)
                pa1 = (T - t) / T
                if np.random.rand() < pa1:
                    r = np.random.rand()
                    for j in range(D):
                        Xm_j = np.mean(positions[:, j])
                        if t <= T / 2.0:
                            if np.random.rand() > np.random.rand():
                                positions[i][j] = Xm_j + RL[i][j] * (positions[A][j] - positions[B][j]) + mu * (r**2 * (ub[j] - lb[j]))
                        else:
                            if np.random.rand() > np.random.rand():
                                positions[i][j] = positions[C][j] + mu * (positions[A][j] - positions[B][j]) + mu * (np.random.rand() < delta) * (r**2 * (ub[j] - lb[j]))
                else:
                    if np.random.rand() < np.random.rand():
                        for j in range(D):
                            positions[i][j] = positions[i][j] + mu * abs(RL[i][j]) * (xbest[j] - positions[i][j]) + np.random.rand() * (positions[A][j] - positions[B][j])
                    elif np.random.rand() < np.random.rand():
                        for j in range(D):
                            positions[i][j] = xbest[j] + mu * (positions[A][j] - positions[B][j])
                    else:
                        for j in range(D):
                            positions[i][j] = xbest[j] * abs(l)

                positions[i] = np.clip(positions[i], lb, ub)
                fit[i] = fobj(positions[i])
                if fit[i] < lfit[i]:
                    lfit[i] = fit[i]
                    lbest[i] = positions[i].copy()
                if fit[i] < best_fit:
                    best_fit = fit[i]
                    xbest = positions[i].copy()
        else:
            for i in range(N):
                theta = np.pi * np.random.rand()
                A, B = np.random.choice(N, 2, replace=False)
                for j in range(D):
                    RP[0][j] = positions[i][j] + alpha * np.cos(theta) * (positions[A][j] - positions[B][j]) if theta != np.pi / 2 else positions[i][j] + alpha * np.cos(theta) * (positions[A][j] - positions[B][j]) + alpha * RP[np.random.randint(2)][j]
                    RP[1][j] = positions[i][j] + alpha * np.cos(theta) * ((ub[j] - lb[j]) * np.random.rand() + lb[j]) * (np.random.rand() < Prp) if theta != np.pi / 2 else positions[i][j] + alpha * np.cos(theta) * ((ub[j] - lb[j]) * np.random.rand() + lb[j]) + alpha * RP[np.random.randint(2)][j] * (np.random.rand() < Prp)
                
                for k in range(2):
                    RP[k] = np.clip(RP[k], lb, ub)

                pa2 = 0.2
                if np.random.rand() > pa2:
                    C = np.random.choice(N)
                    if np.random.rand() < np.random.rand():
                        for j in range(D):
                            if np.random.rand() >= np.random.rand():
                                positions[i][j] = positions[i][j] + np.random.rand() * (xbest[j] - positions[i][j]) + np.random.rand() * (RP[0][j] - positions[C][j])
                    else:
                        for j in range(D):
                            if np.random.rand() >= np.random.rand():
                                positions[i][j] = positions[i][j] + np.random.rand() * (xbest[j] - positions[i][j]) + np.random.rand() * (RP[1][j] - positions[C][j])
                    
                    positions[i] = np.clip(positions[i], lb, ub)
                    fit[i] = fobj(positions[i])
                    if fit[i] < lfit[i]:
                        lfit[i] = fit[i]
                        lbest[i] = positions[i].copy()
                    if fit[i] < best_fit:
                        best_fit = fit[i]
                        xbest = positions[i].copy()
                else:
                    if fobj(RP[1]) < fobj(RP[0]) and fobj(RP[1]) < fit[i]:
                        positions[i] = RP[1].copy()
                        fit[i] = fobj(RP[1])
                    elif fobj(RP[0]) < fobj(RP[1]) and fobj(RP[0]) < fit[i]:
                        positions[i] = RP[0].copy()
                        fit[i] = fobj(RP[0])
                    
                    if fit[i] < lfit[i]:
                        lfit[i] = fit[i]
                        lbest[i] = positions[i].copy()
                    if fit[i] < best_fit:
                        best_fit = fit[i]
                        xbest = positions[i].copy()

        best_fit_per_gen[t-1] = best_fit

    return xbest, best_fit, best_fit_per_gen
