import random
import numpy as np

def initialize_population(N, D, lb, ub):

    """
    Initialize the population.
    
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
    initial_positions = lob + (ub - lb) * (1 - rm)
    
    return initial_positions


def forage_storage(N, D, lb, ub, T, pa1, mu, tau1, tau2, tau3, levy_flight=lambda size: np.random.uniform(-1, 1, size)):

    """
    First stage of NOA - Forage and storage

    Args:
        N (int): Number of individuals in the population.
        D (int): Number of dimensions in the search space.
        lb (numpy.ndarray): Lower bounds for each dimension (Lj).
        ub (numpy.ndarray): Upper bounds for each dimension (Uj).
        T (int): Evaluation times.
        pa1 (float): Probability threshold that decreases over time, controlling exploration vs. exploitation.
        mu (float): Step size scaling factor for exploration.
        tau1, tau2, tau3 (float): Scaling factors for managing exploration ability.
        levy_flight (function): Function to generate Levy-flight steps.

    Returns:
        xbest (numpy.ndarray): The best position found
    """

    # Initialize
    initial_positions = initialize_population(N, D, lb, ub) # Generate N numbers of nutcracker
    xbest = np.min(initial_positions, axis=0) # Initialize the best position as the minimum value across all individuals

    # Begin evolving

    for t in range(0,T):

        pa1 = max(0, pa1 - (1/T)) # Linear decrease of pa1 over time

        for i in range(0,N):
            phi = np.random.randint(0, 2) # Choice factor, assuming 0 or 1 for simplicity
            new_positions = np.zeros(D) # Initialize
            for j in range(0,D):

                if phi > pa1: # Exploration phase 1

                    # Eq. (1)

                    # Calculate the mean of the current population in dimension j
                    Xm_j = np.mean(pop[:, j])

                    # Randomly select three different indices A, B, and C from the population
                    A, B, C = np.random.choice(N, 3, replace=False)

                    gamma = levy_flight(1)[0]  # Generate a random number following the Levy flight distribution
                    r1 = np.random.rand()  # Generate a random number between 0 and 1
                    
                    if tau1 < tau2:
                        new_positions[j]=initial_positions[i][j]

                    elif t<T/2.0: # Global exploration
                        gamma = levy_flight(1)[0]  # Levy flight random number
                        new_positions[j] = Xm_j + gamma * (initial_positions[A][j] - initial_positions[B][j]) + mu * (r**2 * ub - lb)
                    
                    else: # Explore around a random solution with probability δ

                        delta = ...  # Define the value of δ here, possibly from sensitivity analysis

                        # Calculate mu based on Eq. (2)
                        if r1 < r2:
                            mu = tau3
                        elif r2 < r3:
                            mu = tau4 # Normal distribution
                        else:
                            mu = tau5 # Levy flight

                        if r1 < delta:
                            new_positions[j] = initial_positions[C][j] + mu * (initial_positions[A][j] - initial_positions[B][j])  # Local exploration around a chosen solution
                        else:
                            new_positions[j] = initial_positions[C][j] + mu * (initial_positions[A][j] - initial_positions[B][j]) + mu * (r**2*ub-lb)

                else: # Exploitation phase 1 (storage)

                    # Following Eq.(3)
                    lambda_levy = levy_flight(1)[0]
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    if tau1 < tau2:
                        new_positions[j] = initial_positions[i][j] + mu * (xbest[j] - initial_positions[i][j]) * abs(lambda_levy) + r1 * (initial_positions[np.argmin(initial_positions[:, j]), j] - initial_positions[np.argmax(initial_positions[:, j]), j])

                    elif tau1 < tau3:
                        new_positions[j] = xbest[j] + mu * (initial_positions[np.argmin(initial_positions[:, j]), j] - initial_positions[np.argmax(initial_positions[:, j]), j])

                    else:
                        new_positions[j] = xbest[j] * l # 'l' is a linear factor

            # Apply boundary checks to ensure new_positions are within lb and ub
            new_positions = np.clip(new_positions, lb, ub)

            # Update the population and check for new best position
            initial_positions[i] = new_positions
            xbest = np.minimum(xbest, new_positions)
    return xbest



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

    # Initialize
    initial_positions = initialize_population(N, D, lb, ub)  # Generate N nutcrackers
    xbest = np.min(initial_positions, axis=0)  # Initialize the best position

    # Begin evolving
    for t in range(0,T):  # Iterate through generations
        new_positions = np.zeros((N, D))
        pa2 = 0.2 # This value was established from the experiments conducted later

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
                pass
                # Evaluate new positions and update xbest if necessary
                # This step depends on your specific evaluation function and stopping criteria
                # You can use an external function like evaluate_fitness() to calculate the fitness of each individual
                # and then compare it against xbest's fitness to decide whether to update xbest
                # Ensure positions stay within bounds

        # Update the population with new positions
        initial_positions = new_positions
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


    # Stage 1: Forage and Storage Strategy
    xbest_after_forage = forage_storage(N, D, lb, ub, T, pa1, mu, tau1, tau2, tau3)


    # Stage 2: Cache-search and Recovery Strategy
    # Reset population for the second stage or reuse positions from the first stage based on strategy
    # For simplicity, we assume initializing a new population for the second stage
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