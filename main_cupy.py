# This is an EXPERIMENTAL GPU port of original NOA algorithm Python implemention (main.py)
# It uses cupy instead of numpy
# Tested on CUDA 12.4 on RTX 4060 Max-Q (8GB) and works
# Test results show that it is not much faster than original CPU calculation one (or even slower)

import cupy as cp
import opfunu
from cupyx.scipy.special import gamma


# Get CEC testing functions
def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(cp.asnumpy(x))
    return F


def initialization(search_agents_no, dim, ub, lb):
    return cp.random.uniform(lb, ub, (search_agents_no, dim))

# Levy flight function
def levy(n, m, beta):
    num = gamma(1 + beta) * cp.sin(cp.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = cp.random.normal(0, sigma_u, (n, m))
    v = cp.random.normal(0, 1, (n, m))
    z = u / (cp.abs(v) ** (1 / beta))
    return z

def NOA(search_agents_no, max_iter, ub, lb, dim, fobj):
    best_nc = cp.zeros(dim)
    best_score = cp.inf
    lfit = cp.inf * cp.ones(search_agents_no)
    rp = cp.zeros((2, dim))
    convergence_curve = cp.zeros(max_iter)

    alpha = 0.05
    pa2 = 0.2
    prb = 0.2

    positions = initialization(search_agents_no, dim, ub, lb)
    lbest = cp.copy(positions)
    t = 0

    nc_fit = cp.zeros(search_agents_no)
    for i in range(search_agents_no):
        nc_fit[i] = fobj(positions[i, :])
        lfit[i] = nc_fit[i]

        if nc_fit[i] < best_score:
            best_score = nc_fit[i]
            best_nc = positions[i, :]

    while t < max_iter:

        print(f"t = {t}")

        rl = 0.05 * levy(search_agents_no, dim, 1.5) # Levy random number vector
        l = cp.random.rand() * (1 - t / max_iter) # Parameter in Eq. (3)
        
        # Parameter in Eq. (11)
        if cp.random.rand() < cp.random.rand():
            a = (t / max_iter) ** (2 * 1 / (t + 1))
        else:
            a = (1 - (t / max_iter)) ** (2 * (t / max_iter))

        if cp.random.rand() < cp.random.rand(): # Foraging and storage
            mo = cp.mean(positions, axis=0)
           
            # Eq. (2)
            for i in range(search_agents_no):
                if cp.random.rand() < cp.random.rand():
                    mu = cp.random.rand()
                elif cp.random.rand() < cp.random.rand():
                    mu = cp.random.randn()
                else:
                    mu = rl[0, 0]

                cv = cp.random.randint(search_agents_no)
                cv1 = cp.random.randint(search_agents_no)
                pa1 = ((max_iter - t) / max_iter)
                if cp.random.rand() < pa1:
                    cv2 = cp.random.randint(search_agents_no)
                    r2 = cp.random.rand()
                    for j in range(dim):
                        if t < max_iter / 2:
                            if cp.random.rand() > cp.random.rand():
                                positions[i, j] = (mo[j]) + rl[i, j] * (positions[cv, j] - positions[cv1, j]) + mu * (
                                            cp.random.rand() < 5) * (r2 * r2 * ub[j] - lb[j])
                        else:
                            if cp.random.rand() > cp.random.rand():
                                positions[i, j] = positions[cv2, j] + mu * (
                                            positions[cv, j] - positions[cv1, j]) + mu * (cp.random.rand() < alpha) * (
                                                              r2 * r2 * ub[j] - lb[j])
                else:
                    mu = cp.random.rand()
                    if cp.random.rand() < cp.random.rand():
                        r1 = cp.random.rand()
                        for j in range(dim):
                            positions[i, j] = ((positions[i, j])) + mu * abs(rl[i, j]) * (
                                        best_nc[j] - positions[i, j]) + (r1) * (positions[cv, j] - positions[cv1, j])
                    elif cp.random.rand() < cp.random.rand():
                        for j in range(dim):
                            if cp.random.rand() > cp.random.rand():
                                positions[i, j] = best_nc[j] + mu * (positions[cv, j] - positions[cv1, j])
                    else:
                        for j in range(dim):
                            positions[i, j] = (best_nc[j] * abs(l))

                for j in range(dim):
                    if positions[i, j] > ub[j]:
                        positions[i, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])
                    elif positions[i, j] < lb[j]:
                        positions[i, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])

                nc_fit[i] = fobj(positions[i, :])

                if nc_fit[i] < lfit[i]:
                    lfit[i] = nc_fit[i]
                    lbest[i, :] = positions[i, :]
                else:
                    nc_fit[i] = lfit[i]
                    positions[i, :] = lbest[i, :]

                if nc_fit[i] < best_score:
                    best_score = nc_fit[i]
                    best_nc = positions[i, :]

                t += 1
                if t >= max_iter:
                    break
                convergence_curve[t] = best_score

        else:
            for i in range(search_agents_no):
                ang = cp.pi * cp.random.rand()
                cv = cp.random.randint(search_agents_no)
                cv1 = cp.random.randint(search_agents_no)
                for j in range(dim):
                    for j1 in range(2):
                        if j1 == 1:
                            if ang != cp.pi / 2:
                                rp[j1, j] = positions[i, j] + (a * cp.cos(ang) * (positions[cv, j] - positions[cv1, j]))
                            else:
                                rp[j1, j] = positions[i, j] + a * cp.cos(ang) * (
                                            positions[cv, j] - positions[cv1, j]) + a * rp[cp.random.randint(2), j]
                        else:
                            if ang != cp.pi / 2:
                                rp[j1, j] = positions[i, j] + (a * cp.cos(ang) * ((ub[j] - lb[j]) + lb[j])) * (
                                            cp.random.rand() < prb)
                            else:
                                rp[j1, j] = positions[i, j] + (
                                            a * cp.cos(ang) * ((ub[j] - lb[j]) * cp.random.rand() + lb[j]) + a * rp[
                                        cp.random.randint(2), j]) * (cp.random.rand() < prb)

                for j in range(dim):
                    if rp[1, j] > ub[j]:
                        rp[1, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])
                    elif rp[1, j] < lb[j]:
                        rp[1, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])

                for j in range(dim):
                    if rp[0, j] > ub[j]:
                        rp[0, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])
                    elif rp[0, j] < lb[j]:
                        rp[0, j] = lb[j] + cp.random.rand() * (ub[j] - lb[j])

                if cp.random.rand() < pa2:
                    cv = cp.random.randint(search_agents_no)
                    if cp.random.rand() < cp.random.rand():
                        for j in range(dim):
                            if cp.random.rand() > cp.random.rand():
                                positions[i, j] = positions[i, j] + cp.random.rand() * (
                                            best_nc[j] - positions[i, j]) + cp.random.rand() * (
                                                              rp[0, j] - positions[cv, j])
                    else:
                        for j in range(dim):
                            if cp.random.rand() > cp.random.rand():
                                positions[i, j] = positions[i, j] + cp.random.rand() * (
                                            best_nc[j] - positions[i, j]) + cp.random.rand() * (
                                                              rp[1, j] - positions[cv, j])
                else:
                    cv = cp.random.randint(search_agents_no)
                    if cp.random.rand() < cp.random.rand():
                        for j in range(dim):
                            positions[i, j] = positions[i, j] + cp.random.rand() * (
                                        best_nc[j] - positions[i, j]) + a * (positions[cv, j] - positions[cv1, j])
                    else:
                        for j in range(dim):
                            positions[i, j] = positions[i, j] + cp.random.rand() * (best_nc[j] - positions[i, j])

                nc_fit[i] = fobj(positions[i, :])

                if nc_fit[i] < lfit[i]:
                    lfit[i] = nc_fit[i]
                    lbest[i, :] = positions[i, :]
                else:
                    nc_fit[i] = lfit[i]
                    positions[i, :] = lbest[i, :]

                if nc_fit[i] < best_score:
                    best_score = nc_fit[i]
                    best_nc = positions[i, :]

                t += 1
                if t >= max_iter:
                    break
                convergence_curve[t] = best_score


    return best_nc, best_score, convergence_curve

# Parameters
SearchAgents_no = 200 # Number of nutcrackers
MaxFES = 1000000 # Evaluation times
RUN_NO = 5 # Run how many times

# Function numbers
# Specify a list to select which functions to run
# For CEC-2014 the list is from 1 to 30
Fun_id = list(range(1, 31))
# For CEC-2017 it is from 1 to 29
# func_ids=range(1,30)
# For CEC-2020 it is from 1 to 10
# func_ids=range(1,11)

# DEBUG: Just to test if the NOA algorithm is correct
def fobj(x):
    return cp.sum(x ** 2)

# Calculate the average best for each generation
average_convergence_curve = cp.zeros(MaxFES)

for i in range(1,31):
    results = []
    for j in range(RUN_NO):
        fun_name = f'F{i}'
        # CEC year controlling parameters
        # Available: 2014, 2017, 2020
        year = '2014'
        func_num = fun_name + year
        dim = 30 # Dimension of the question
        lb = -100 * cp.ones(dim) # Lower boundary
        ub = 100 * cp.ones(dim) # Upper boundary
        cec = 1 # Whether to use CEC function
        if cec == 0:
            fun = fobj
            func_num = 'fobj'
        else:
            fun = cec_fun

        print(f"Function {i} of CEC-{year} begins for {j+1} time(s)")
        best_position, best_score, convergence_curve = NOA(SearchAgents_no, MaxFES, ub, lb, dim, fun)
        average_convergence_curve += convergence_curve
        results.append(convergence_curve[-1])
    average_convergence_curve /= (30 * RUN_NO)

        # If "results/" does not exist, create it
    if not os.path.exists("results"):
        os.makedirs("results")

    # Save the average best into the file
    with open(f'results/{func_num}_process.txt', 'w') as f:
        for value in average_convergence_curve:
            f.write(f'{value}\n')

    # Save the result
    with open(f'results/{func_num}_results.txt', 'w') as f:
        for value in results:
            f.write(f'{value}\n')

    print(f'{func_num} saved to results/{func_num}_results.txt')
