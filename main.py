from NOA import nutcracker_optimizer
import opfunu
import numpy as np
import os

# Controlling parameters
N = 50 # Number of nutcrackers
T = 500 # Evaluation times
delta = 0.05 # The percent of attempts at avoiding local optima
Pa2 = 0.2 # The probability of exchanging between the cache-search stage and the recovery stage
Prp = 0.2 # The percentage of exploration other regions within the search space
run_time=1 # Run how many times

# CEC-specific controlling parameters
cec_year = 2014 # Run which CEC function

# Function number, from 1 to 30
# Specify a list to select which functions to run
func_ids=[1]


# 运行 NOA 并保存结果
def run_and_save_results(func_num, fobj, lb, ub, dim, num_runs):
    all_runs_best_per_gen = []
    all_runs_final_best = []
    for i in range(1,num_runs+1):
        print(f'{func_num} 第 {i} 次开始')
        X_best, f_best, f_best_per_gen = nutcracker_optimizer(N, D, lb, ub, T, delta, Prp, fobj)
        all_runs_best_per_gen.append(f_best_per_gen)
        all_runs_final_best.append(f_best_per_gen[-1])  # 添加每次运行的最后结果

    # 计算每代的平均最优值
    avg_best_per_gen = np.mean(all_runs_best_per_gen, axis=0)

    # If "results/" does not exist, create it
    if not os.path.exists("results"):
        os.makedirs("results")

    # 保存平均过程到文件
    with open(f'results/average_{func_num}.txt', 'w+') as f:
        for t, avg_best in enumerate(avg_best_per_gen):
            f.write(f'{avg_best}\n')

    # 保存每次结果
    with open(f'results/raw_{func_num}.txt', 'w+') as f:
        for final_best in all_runs_final_best:
            f.write(f'{final_best}\n')
        f.write(f'{avg_best_per_gen[-1]}\n')

# 定义测试函数 ftest
def ftest(x):
    return np.sum(x ** 2)

# 定义的 cec 函数
def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=D)
    F = func.evaluate(x)
    return F


# Main entrance
if __name__ == '__main__':

    # For functions in CEC2014, these are fixed
    D = 30
    lb = -100 * np.ones(D)
    ub = 100 * np.ones(D)
    

    for func_id in func_ids:
        fun_name = 'F' + f'{func_id}'
        year = '2014'
        func_num = fun_name + year
        # run_and_save_results("ftest", ftest, lb, ub, D, run_time)
        run_and_save_results(func_num, cec_fun, lb, ub, D, run_time)
