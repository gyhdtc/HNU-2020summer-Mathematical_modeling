from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
# 协方差矩阵
F = np.array([[4,5/2,-10],
              [5/2,36,-15],
              [-10,-15,100]])
m1 = np.array([20, 25, 30])
m2 = np.array([25,33,40])
m3 = np.array([5,8,10])
beta = 0
# 问题1优化目标函数
def func(x):
    return x @ F @ x.T

# 问题一求解
def investment():
    con1 ={'type': 'ineq', 'fun': lambda x: 500000-m1@x}
    con2 ={'type': 'ineq', 'fun': lambda x: m3@x-100000}
    b = (0, 500000)
    bound = (b, b, b)
    res = minimize(func, np.ones(3)+1e-5, constraints=[con1,con2],bounds=bound)
    print(res) #输出解的信息
    print('投资:',m1 @ res['x'])
    print("收益:",m3 @ res['x'])
    print('最优值:',func(res['x']))
    print('最优解：',res['x'])


# 求解固定beta时问题二的目标函数最优值
def cov():
    con1 = {'type': 'ineq', 'fun': lambda x: 500000 - m1 @ x}
    b = (1,500000)
    bound = (b,b,b)
    res = minimize(func2, np.ones(3) + 1e-5, constraints=con1,bounds=bound)
    return m2 @ res['x'] - 500000,-res['x']@F@res['x']
# 问题二优化目标函数
def func2(x):
    return beta*(x @ F @ x.T)- (m2@x)


if __name__ == '__main__':
    investment()
    income_list = []
    risk_list = []
    # 画出关系图
    for i in np.arange(0,100,1):
        beta=i*0.0000001
        income,risk = cov()
        income_list.append(income)
        risk_list.append(risk)
    plt.subplot(221)
    x = np.arange(0,100,1)
    plt.plot(x,income_list,color='green')
    plt.xlabel('beta/1e-8')
    plt.ylabel('income')
    plt.subplot(222)
    plt.xlabel('beta/1e-8')
    plt.ylabel('-risk')
    plt.plot(x,risk_list,color='red')
    plt.subplot(212)
    plt.xlabel('-risk')
    plt.ylabel('income')
    plt.plot(risk_list,income_list,color='yellow')
    plt.show()