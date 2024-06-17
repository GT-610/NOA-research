import random
import numpy as np

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

def cache_recovery(N, D, lb, ub, Tmax, gamma, mu, tau7, tau8, delta, RP_matrix):
    """
    Second stage of NOA - Cache-search and recovery strategy

    Args:
        N (int): Number of nutcrackers in the population.
        D (int): Number of dimensions in the search space.
        lb (numpy.ndarray): Lower bounds for each dimension.
        ub (numpy.ndarray): Upper bounds for each dimension.
        Tmax (int): Maximum number of generations.
        gamma, mu (float): Control parameters for updating nutcracker positions.
        r1, r2 (float): Random numbers for strategy decisions.
        delta (float): Threshold for exploration decision.
        RP_matrix (numpy.ndarray): Matrix of reference points for each cache.

    Returns:
        xbest (numpy.ndarray): The best position found after the recovery phase.
    """

    # Begin evolving
    new_positions = np.zeros((N, D))
    pa2 = 0.2 # This value was established from the experiments conducted later. DO NOT CHANGE

    for i in range(N):  # For each nutcracker
        phi = np.random.randint(0, 2)
        if phi > pa2: # Exploration phase 2

            # Cache-search stage
            if tau7 < tau8:  # The nutcracker remembers the hidden cache position
                for j in range(D):  # For each dimension
                # Eq. (16)
                    # Eq. (13)
                    if r3 < r4:  
                        new_positions[i][j] = initial_positions[i][j]
                    else:
                        new_positions[i][j] = initial_positions[i][j] + r1 * (xbest[j] - initial_positions[i][j]) + r2 * (RP_matrix[i][j] - initial_positions[i][j])

            else:  # The nutcracker forgets the hidden cache position
                for j in range(D):  # For each dimension
                    # Eq. (15)
                    if r5 < r6:  
                        new_positions[i][j] = initial_positions[i][j]
                    else:
                        new_positions[i][j] = initial_positions[i][j] + r1 * (xbest[j] - initial_positions[i][j]) + r2 * (RP_matrix[i][j+D] - initial_positions[i][j])

            new_positions = np.clip(new_positions, lb, ub) # Border check


        else: # Exploitation phase 2
            for j in range(D):

            # Introduce a local exploitation mechanism
                r1 = np.random.rand()
                r2 = np.random.rand()

            # If the random value r1 is less than a certain threshold (e.g., delta), perform a local search around xbest
                if r1 < delta:
                    # Use a combination of the global best and a local perturbation
                       new_positions[i][j] = xbest[j] + mu * (r2 * (ub[j] - lb[j]))  # Local perturbation with control parameter mu

                else:
                    # Otherwise, rely on the cache memory (RP_matrix) but with a higher focus on the best solution
                    if tau7 < tau8:  # Favoring remembered cache positions
                        new_positions[i][j] = xbest[j] + r1 * (RP_matrix[i][j] - xbest[j])  # Bias towards known good positions

                    else:  # If Pa1cache memory isn't favorable, explore more conservatively
                        new_positions[i][j] = xbest[j] + mu * ((RP_matrix[i][j] + RP_matrix[i][j+D])/2 - xbest[j])  # Average of both RPs for保守探索

            # Ensure the positions stay within the defined boundaries
            new_positions[Pa1i] = np.clip(new_positions[i], lb, ub)

        # After updating positions for all nutcrackers, evaluate their fitness
        # Assuming a function evaluate_fitness is defined elsewhere
        fitness_scores = evaluate_fitness(new_positions)

        # Update the global best solution
        best_fitness_idx = np.argmax(fitness_scores) if maximizing_problem else np.argmin(fitness_scores)
        xbest = new_positions[best_fitness_idx]
        
        # Update the population with the new positions
        initial_positions = np.copy(new_positions)             
        xbest = np.minimum(xbest, new_positions)
    return xbest



def nutcracker_optimizer(N, D, lb, ub, T, Tmax, pa1, mu, tau1, tau2, tau3, delta, RP_matrix):

    """
    Full Nutcracker Optimization Algorithm (NOA) process integrating both foraging-storage and cache-search-recovery strategies.

    Args:
        N, D, lb, ub (int, int, numpy.ndarray, numpy.ndarray): Same as previous definitions.
        T (int): Evaluation times for the foraging-storage strategy.
        Tmax (int): Maximum number of generations for the entire process.
        pa1 (float): Probability threshold for switching between exploration and exploitation in the first strategy.
        mu, tau1, tau2, tau3 (float): Control parameters for the first strategy.
        delta (float): Threshold for exploration decision in the second strategy.
        RP_matrix (numpy.ndarray): Matrix of reference points for the cache-search strategy.

    Returns:
        xbest_final (numpy.ndarray): The best position found after the entire NOA process.
    """

    # Initialize
    positions = initialize_population(N, D, lb, ub) # Generate N numbers of nutcracker
    # Initialize the best position
    # At the beginning of the algorithm, the current position of each Nutcracker (i.e., search agent) is set to its respective local best solution (Lbest).
    # This means that the starting position of each individual is considered to be its current best solution.
    xbest = np.min(positions, axis=0)

    for t in range(0,T):

        RL = 0.05*levy(N,dim,1.5) # Levy flight
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

                pa1 = ((T-t))/T

                for j in range(0,D): # For every dimension
                    A, B, C = np.random.choice(N, 3, replace=False)
                    if np.random.rand() < pa1: # Exploration phase 1. phi < pa1 FIXME
                        Xm_j = np.mean(positions[:, j])
                        if np.random.rand() >= np.random.rand():
                            if t<=T/2.0: # Global exploration
                                positions[i][j] = Xm_j + RL[i][j] * (initial_positions[A][j] - initial_positions[B][j]) + mu * (r**2 * ub - lb)
                    
                            else: # Explore around a random solution with probability δ
                                positions[i][j] = positions[C][j] + mu * (np.random.rand() < delta) * (initial_positions[A][j] - initial_positions[B][j])  # Local exploration around a chosen solution
                    
                    else: # Exploitation phase 1 (storage)
                        # Following Eq.(3)
                        if np.random.rand() < np.random.rand(): # tau1 < tau2
                            positions[i][j] = positions[i][j] + mu * (xbest[j] - positions[i][j]) * abs(RL[i][j]) + r1 * (positions[A][j]-positions[B][j])
                        elif np.random.rand() < np.random.rand(): # tau1 < tau3 
                            positions[i][j] = xbest[j] + mu * (positions[A][j]-positions[B][j])
                        else:
                            positions[i][j] = xbest[j] * l

                    # Boundary check

        else:
            # Stage 2: Cache-search and Recovery Strategy

            initial_positions_for_recovery = initialize_population(N, D, lb, ub)
            xbest_final = cache_recovery(N, D, lb, ub, Tmax, mu, tau1, tau2, delta, RP_matrix)

    # Return the best solution found across both stages
    # Note: Depending on the specific implementation details, you might want to compare solutions from both stages
    return np.minimum(xbest_after_forage, xbest_final)


# Example usage:

# N, D, lb, ub, T, Tmax, pa1, mu, tau1, tau2, tau3, delta = ... # Define your parameters here

# RP_matrix = ... # Define your reference points matrix here

# best_solution = nutcracker_optimizer(N, D, lb, ub, T, Tmax, pa1, mu, tau1, tau2, tau3, delta, RP_matrix)

# print("Best solution found:", best_solution)