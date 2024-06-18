import random
import cupy as cp
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
    rm = cp.random.rand(N, D)
    
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
    xbest = cp.min(positions, axis=0)# Best nutcracker position (solution)
    best_fit=cp.inf # Best fitness
    lfit=cp.inf * cp.ones(N) # Local-best fitness for each Nutcracker
    lbest = positions # Local-best solution for each Nutcracker as its current position at the beginning
    # 2D matrix to include two reference points
    RP=cp.zeros((2,D))
    fit = cp.inf * cp.ones(N) # Current fitness of every nutcracker

    # --- Evaluate ---
    for i in range(0,N):
        fit[i]=fobj(positions[i])
        lfit[i]=fit[i] # Set the local best fitness for the ith Nutcracker as its current fitness

        # Update the best solution
        if fit[i]<best_fit:
            best_fit=fit[i]
            xbest = positions[i]

    # Store every best fitness of each evolution
    best_fit_per_gen = cp.zeros(T)

    # --- Begin ---
    for t in range(1,T+1):

        RL = 0.05*levy(N,D,1.5) # Levy flight
        # Parameter in Eq.(3)
        l = cp.random.rand()*(1-t/T)
        # Paramater in Eq. (11)
        if cp.random.rand()>cp.random.rand():
            alpha = (1-t/T)**(2*1/t)
        else:
            alpha = (t/T)**(2/t)
        
        # Stage 1: Forage and Storage Strategy 
        if cp.random.rand() < cp.random.rand(): # sigma < sigma1

            for i in range(0,N): # For every nutcracker
                # Calculate mu based on Eq. (2)
                if cp.random.rand() < cp.random.rand(): # r1 < r2
                    mu = cp.random.rand() # tau3
                elif cp.random.rand() < cp.random.rand(): # r2 < r3 
                    mu = cp.random.randn() # tau4
                else:
                    mu = RL[0][0] # Levy flight

                A, B, C = cp.random.choice(N, 3, replace=False)
                pa1 = ((T-t))/T

                if cp.random.rand() < pa1: # Exploration phase 1. phi < pa1 FIXME
                    r = cp.random.rand()

                    for j in range(0,D): # For every dimension
                        Xm_j = cp.mean(positions[:, j])

                        # Eq. (1)
                        if cp.random.rand() >= cp.random.rand(): # tau1 > tau2
                            if t<=T/2.0: # Global exploration
                                positions[i][j] = Xm_j + RL[i][j] * (positions[A][j] - positions[B][j]) + mu * (r**2 * ub[j] - lb[j])
                    
                            else: # Explore around a random solution with probability δ
                                positions[i][j] = positions[C][j] + mu * (positions[A][j] - positions[B][j]) + mu * (cp.random.rand() < delta) * (r**2 * ub[j] - lb[j])  # Local exploration around a chosen solution
                    
                else: # Exploitation phase 1
                    # Following Eq.(3)
                    if cp.random.rand() < cp.random.rand(): # tau1 < tau2
                        positions[i][j] = positions[i][j] + mu * (xbest[j] - positions[i][j]) * abs(RL[i][j]) + cp.random.rand() * (positions[A][j]-positions[B][j])
                    elif cp.random.rand() < cp.random.rand(): # tau1 < tau3 
                        positions[i][j] = xbest[j] + mu * (positions[A][j]-positions[B][j])
                    else:
                        positions[i][j] = xbest[j] * l

                # Border check for nutcrackers
                if cp.random.rand() < cp.random.rand():
                    for j in range(0,D):
                        if positions[i][j]>ub[j] or positions[i][j]<lb[j]:
                            positions[i][j] = lb[j]+cp.random.rand() * (ub[j]-lb[j])

                fit[i] = fobj(positions[i])

                # Update local best acording to Eq. (20)
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

        else:
            # Stage 2: Cache-search and Recovery Strategy
            # Generate RPs for each nutcracker
            for i in range(N):
                theta = cp.pi * cp.random.rand()
                A, B = cp.random.choice(N, 2, replace=False)
                for j in range(0,D):
                    # The first RP, following Eq. (9)
                    if theta != cp.pi / 2:
                        RP[0][j] = positions[i][j] + (alpha * cp.cos(theta) * (positions[A][j]-positions[B][j]))
                    else:
                        RP[0][j] = positions[i][j] + (alpha * cp.cos(theta) * (positions[A][j]-positions[B[j]])) + alpha * RP[cp.random.randint(2)][j]
                    
                    # The second RP, following Eq. (10)
                    if theta != cp.pi / 2:
                        RP[1][j] = positions[i][j] + (alpha * cp.cos(theta) * ((ub[j] - lb[j]) * cp.random.rand() + lb[j])) * (cp.random.rand() < Prp) # No definations for U2; Might be U1 in the article
                    else:
                        RP[1][j] = positions[i][j] + (alpha * cp.cos(theta) * ((ub[j] - lb[j]) * cp.random.rand() + lb[j]) + alpha * RP[cp.random.randint(2)][j]) * (cp.random.rand() < Prp)
                
                # Exceed RP return
                for k in range(0,2):
                    if cp.random.rand() < cp.random.rand():
                        for j in range(0,D):
                            if RP[k][j]>ub[j] or RP[k][j]<lb[j]:
                                RP[k][j]=lb[j]+cp.random.rand()*(ub[j]-lb[j])

                pa2 = 0.2 # This value was established from the experiments conducted later. DO NOT CHANGE
                if cp.random.rand() > pa2: # Exploration phase 2 (cache-search)
                    # Eq. (16)
                    C = cp.random.choice(N,size=1)[0]
                    if cp.random.rand() < cp.random.rand(): # tau7 >= tau8
                        # Eq. (13)
                        for j in range(0,D):
                            if cp.random.rand() >= cp.random.rand(): #tau3 >= tau4
                                positions[i][j] = positions[i][j] + cp.random.rand() * (xbest[j] - positions[i][j]) + cp.random.rand() * (RP[0][j] - positions[C][j])
                    else:
                        # Eq. (15)
                        for j in range(0,D):
                            if cp.random.rand() >= cp.random.rand(): # tau5 >= tau6
                                positions[i][j] = positions[i][j] + cp.random.rand() * (xbest[j] - positions[i][j]) + cp.random.rand() * (RP[1][j] - positions[C][j])
                        
                    # Border check for nutcrackers
                    if cp.random.rand() < cp.random.rand():
                        for j in range(0,D):
                            if positions[i][j]>ub[j] or positions[i][j]<lb[j]:
                                positions[i][j] = lb[j]+cp.random.rand() * (ub[j]-lb[j])

                    # Evaluate for one nutcracker
                    fit[i]=fobj(positions[i])

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

                # Update local best acording to Eq. (20)
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
        
        # Store every best fitness
        best_fit_per_gen[t-1]=best_fit

    return xbest, best_fit, best_fit_per_gen