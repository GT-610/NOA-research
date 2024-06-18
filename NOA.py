import random
import numpy as np
from levy import levy

def initialize_population(N, D, lb, ub): # Tested OK

    """
    Args:
        N (int): Number of individuals in the population.
        D (int): Number of dimensions in the search space.
        lb (numpy.ndarray): Lower bounds for each dimension (Lj).
        ub (numpy.ndarray): Upper bounds for each dimension (Uj).
        
    Returns:
        numpy.ndarray: Initial positions of the population (X_it,j) in the search space.
    """

    # Using Eq. (19)
    # Generate a random matrix for each individual with dimensions (N, D)
    rm = np.random.rand(N, D)
    
    # Calculate the scaled positions using Eq.(19)
    initial_positions = lb + (ub - lb) * (1 - rm)
    
    return initial_positions

def nutcracker_optimizer(N, D, lb, ub, T, delta, Prp, fobj):

    """
    Full Nutcracker Optimization Algorithm (NOA) process integrating both foraging-storage and cache-search-recovery strategies.

    Args:
        N, D, lb, ub (int, int, numpy.ndarray, numpy.ndarray): Same as previous definitions.
        T (int): Maximum number of generations for the entire process.
        delta (float): Threshold for exploration decision in the second strategy.
        Prp (int) : a probability employed to determine the percentage of globally
exploring other regions within the search space.
        fobj (function): Test function.
    Returns:
        xbest_final (numpy.ndarray): The best position found after the entire NOA process.
    """

    # --- Initialize ---
    positions = initialize_population(N, D, lb, ub) # Generate N numbers of nutcracker
    xbest = np.min(positions, axis=0)# Best nutcracker position (solution)
    best_fit=np.inf # Best fitness
    lfit=np.inf * np.ones(N) # Local-best fitness for each Nutcracker
    lbest = positions # Local-best solution for each Nutcracker as its current position at the beginning
    # 2D matrix to include two reference points
    RP=np.zeros((2,D))
    fit = np.inf * np.ones(N) # Current fitness of every nutcracker

    # --- Evaluate ---
    for i in range(0,N):
        fit[i]=fobj(positions[i])
        lfit[i]=fit[i] # Set the local best fitness for the ith Nutcracker as its current fitness

        # Update the best solution
        if fit[i]<best_fit:
            best_fit=fit[i]
            xbest = positions[i]

    # --- Begin ---
    for t in range(1,T+1):

        RL = 0.05*levy(N,D,1.5) # Levy flight
        # Parameter in Eq.(3)
        l = np.random.rand()*(1-t/T)
        # Paramater in Eq. (11)
        if np.random.rand()>np.random.rand():
            alpha = (1-t/T)**(2*1/t)
        else:
            alpha = (t/T)**(2/t)
        
        # Stage 1: Forage and Storage Strategy 
        if np.random.rand() < np.random.rand(): # sigma < sigma1

            for i in range(0,N): # For every nutcracker

                # Calculate mu based on Eq. (2)
                if np.random.rand() < np.random.rand(): # r1 < r2
                    mu = np.random.rand() # tau3
                elif np.random.rand() < np.random.rand(): # r2 < r3 
                    mu = np.random.randn() # tau4
                else:
                    mu = RL[0][0] # Levy flight

                A, B, C = np.random.choice(N, 3, replace=False)
                pa1 = ((T-t))/T

                if np.random.rand() < pa1: # Exploration phase 1. phi < pa1 FIXME
                    r = np.random.rand()

                    for j in range(0,D): # For every dimension
                        Xm_j = np.mean(positions[:, j])

                        # Eq. (1)
                        if np.random.rand() >= np.random.rand():
                            if t<=T/2.0: # Global exploration
                                positions[i][j] = Xm_j + RL[i][j] * (positions[A][j] - positions[B][j]) + mu * (r**2 * ub[j] - lb[j])
                    
                            else: # Explore around a random solution with probability δ
                                positions[i][j] = positions[C][j] + mu * (positions[A][j] - positions[B][j]) + mu * (np.random.rand() < delta) * (r**2 * ub[j] - lb[j])  # Local exploration around a chosen solution
                    
                    else: 
                        # Following Eq.(3)
                        if np.random.rand() < np.random.rand(): # tau1 < tau2
                            positions[i][j] = positions[i][j] + mu * (xbest[j] - positions[i][j]) * abs(RL[i][j]) + np.random.rand() * (positions[A][j]-positions[B][j])
                        elif np.random.rand() < np.random.rand(): # tau1 < tau3 
                            positions[i][j] = xbest[j] + mu * (positions[A][j]-positions[B][j])
                        else:
                            positions[i][j] = xbest[j] * l

                    # Boundary check

        else:
            # Stage 2: Cache-search and Recovery Strategy
            # Generate RPs for each nutcracker
            for i in range(N):
                theta = np.pi * np.random.rand()
                A, B = np.random.choice(N, 2, replace=False)
                for j in range(0,D):
                    # The first RP, following Eq. (9)
                    if theta != np.pi / 2:
                        RP[0][j] = positions[i][j] + (alpha * np.cos(theta) * (positions[A][j]-positions[B][j]))
                    else:
                        RP[0][j] = positions[i][j] + (alpha * np.cos(theta) * (positions[A][j]-positions[B[j]])) + alpha * RP[np.random.randint(2)][j]
                    
                    # The second RP, following Eq. (10)
                    if theta != np.pi / 2:
                        RP[1][j] = positions[i][j] + (alpha * np.cos(theta) * ((ub[j] - lb[j]) * np.random.rand() + lb[j])) * (np.random.rand() < Prp) # No definations for U2; Might be U1 in the article
                    else:
                        RP[1][j] = positions[i][j] + (alpha * np.cos(theta) * ((ub[j] - lb[j]) * np.random.rand() + lb[j]) + alpha * RP[np.random.randint(2)][j]) * (np.random.rand() < Prp)
                
                # Exceed RP return
                for k in range(0,2):
                    if np.random.rand() < np.random.rand():
                        for j in range(0,D):
                            if RP[k][j]>ub[j] or RP[k][j]<lb[j]:
                                RP[k][j]=lb[j]+np.random.rand()*(ub[j]-lb[j])

                pa2 = 0.2 # This value was established from the experiments conducted later. DO NOT CHANGE
                if np.random.randint(0, 2) > pa2: # Exploration phase 2 (cache-search)
                    # Eq. (16)
                    C = np.random.choice(N)
                    if np.random.rand() < np.random.rand(): # tau7 >= tau8
                        # Eq. (13)
                        for j in range(0,D):
                            if np.random.rand() >= np.random.rand(): #tau3 >= tau4
                                positions[i][j] = positions[i][j] + np.random.rand() * (xbest[j] - positions[i][j]) + np.random.rand() * (RP[0][j] - positions[C][j])
                    else:
                        # Eq. (15)
                        for j in range(0,D):
                            if np.random.rand() >= np.random.rand(): # tau5 >= tau6
                                positions[i][j] = positions[i][j] + np.random.rand() * (xbest[j] - positions[i][j]) + np.random.rand() * (RP[1][j] - positions[C][j])
                        
                    # Border check for nutcrackers
                    if np.random.rand() < np.random.rand():
                        for j in range(0,D):
                            if positions[i][j]>ub[j] or positions[i][j]<lb[j]:
                                positions[i][j] = lb[j]+np.random.rand() * (ub[j]-lb[j])

                    # Evaluate for one nutcracker
                    fit[i]=fobj(positions[i])

                    # Update local best
                    if fit[i]<lfit[i]: # Change this to > for calculating maximization
                        lfit[i] = fit[i]
                        lbest[i] = positions[i]
                    else:
                        fit[i]=lfit[i]
                        positions[i] = lbest[i]

                    # Update global best
                    if fit[i] < best_fit:
                        best_fit = fit[i]
                        xbest - positions[i]

                else: # Exploitation phase 2 (recovery)
                    # Evaluation
                    fit0 = fobj(RP[0])
                    fit1 = fobj(RP[1])

                    # Judge which RP is closer to the cache
                    # Eq. (17)
                    if fit0 < fit1 and fit0 < fit[i]:
                        positions[i] = RP[0]
                        fit[i] = fit0
                    elif fit1 < fit0 and fit1 < fit[i]:
                        positions[i] = RP[1]
                        fit[i] = fit1

                    # Update local best
                    if fit[i]<lfit[i]: # Change this to > for calculating maximization
                        lfit[i] = fit[i]
                        lbest[i] = positions[i]
                    else:
                        fit[i]=lfit[i]
                        positions[i] = lbest[i]

                    # Update global best
                    if fit[i] < best_fit:
                        best_fit = fit[i]
                        xbest = positions[i]


    return xbest, best_fit, 