import os
import numpy as np
import opfunu

# 创建 results 文件夹
if not os.path.exists('results'):
    os.makedirs('results')


# 定义测试函数 fobj
def fobj(x):
    return np.sum(x ** 2)


'''定义的 cec 函数'''
def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(x)
    return F


''' fit_func->目标函数, lb->下限, ub->上限 '''
problem_dict = {
    "func_num": 'fobj',
    "fit_func": fobj,
}


# 定义 NOA 函数
def NOA(func, lb, ub, dim, N=50, T=500):
    X_dec = np.random.uniform(lb, ub, (N, dim))
    X_new = np.zeros((N, dim))
    f_dec = np.zeros(N)
    f_best = float('inf')
    X_best = np.zeros(dim)

    # 计算种群函数值
    for i in range(N):
        f_dec[i] = func(X_dec[i, :])
        if f_dec[i] < f_best:
            f_best = f_dec[i]
            X_best = X_dec[i, :].copy()

    # 存储每代的最优值
    f_best_per_gen = []

    # 主循环
    t = 0
    while t < T:
        r1, r2 = np.random.rand(), np.random.rand()
        if r1 < r2:
            for i in range(N):
                for j in range(dim):
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()
                    X_m = np.mean(X_dec, axis=0)  # 种群的均值
                    X_A = X_dec[np.random.randint(0, N), :]
                    X_B = X_dec[np.random.randint(0, N), :]
                    X_new[i, j] = X_m[j] + r3 * (X_A[j] - X_B[j]) + r4 * r5 * (ub - lb)
        else:
            for i in range(N):
                for j in range(dim):
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    X_A = X_dec[np.random.randint(0, N), :]
                    X_B = X_dec[np.random.randint(0, N), :]
                    X_new[i, j] = X_dec[i, j] + r3 * (X_best[j] - X_dec[i, j]) + r4 * (X_A[j] - X_B[j])

        # 边界检查
        X_new = np.clip(X_new, lb, ub)

        # 更新个体位置和函数值
        for i in range(N):
            if func(X_new[i, :]) < f_dec[i]:
                X_dec[i, :] = X_new[i, :].copy()
                f_dec[i] = func(X_new[i, :])
                if f_dec[i] < f_best:
                    f_best = f_dec[i]
                    X_best = X_new[i, :].copy()

        # 存储每代的全局最优值
        f_best_per_gen.append(f_best)
        t += 1

    return X_best, f_best, f_best_per_gen


# 运行 NOA 并保存结果
def run_and_save_results(func_num, func, lb, ub, dim, num_runs):
    all_runs_best_per_gen = []
    all_runs_final_best = []
    for i in range(num_runs):
        print(f'{func_num} 第 {i} 次开始')
        X_best, f_best, f_best_per_gen = NOA(func, lb, ub, dim)
        all_runs_best_per_gen.append(f_best_per_gen)
        all_runs_final_best.append(f_best_per_gen[-1])  # 添加每次运行的最后结果



    # 计算每代的平均最优值
    avg_best_per_gen = np.mean(all_runs_best_per_gen, axis=0)

    # 保存平均过程到文件
    with open(f'results/average_{func_num}.txt', 'w') as f:
        for t, avg_best in enumerate(avg_best_per_gen):
            f.write(f'{avg_best}\n')

    # 保存每次结果
    with open(f'results/raw_{func_num}.txt', 'w') as f:
        for final_best in all_runs_final_best:
            f.write(f'{final_best}\n')
        f.write(f'{avg_best_per_gen[-1]}\n')







# 主程序入口
if __name__ == '__main__':
    num_runs = 5
    if_cec = 1
    if if_cec:
        for i in range(1, 31):
            fun_name = 'F' + f'{i}'
            year = '2014'
            func_num = fun_name + year
            dim = 30  # 维度，根据 cec 函数选择对应维度
            run_and_save_results(func_num, cec_fun, -100, 100, dim, num_runs)
    else:
        dim = 10
        run_and_save_results('fobj', fobj, -100, 100, dim, num_runs)
