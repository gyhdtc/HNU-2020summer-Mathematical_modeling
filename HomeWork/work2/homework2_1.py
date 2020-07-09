import numpy as np
import cvxpy as cp
import pandas as pd

# 构造目标函数
# 1. 定义决策向量x(i, j)表示第j个配送中心向第i个部队用户运输的
# 物资量
x = cp.Variable((15, 8), pos = True)
# 2. 定义价值向量(单位物资的运费)
c = np.genfromtxt('../data/homework2_data.txt', dtype = \
       float, max_rows = 15, usecols = range(8))
# 3. 定义目标函数
obj = cp.Minimize(cp.sum(cp.multiply(c, x)))

# 定义约束条件
# 获取需求量
a = np.genfromtxt('../data/homework2_data.txt', dtype= \
       float, max_rows = 15, usecols= 8)
# 获取储备量
b = np.genfromtxt('../data/homework2_data.txt', dtype = \
       float, skip_header = 15) # 读最后一行数量
# 定义约束条件  储备量约束 + 需求量约束
con1 = [cp.sum(x, axis = 1, keepdims = True) == \
        a.reshape(15, 1),
       cp.sum(x, axis = 0, keepdims = True) <= \
        b.reshape(1, 8)]

## 求解问题
prob = cp.Problem(obj, con1)
prob.solve(solver = 'GLPK_MI')
print('最优值', obj.value)
print('最优解', x.value)

xd = pd.DataFrame(x.value, dtype = int)
xd.to_excel('../data/homework2_res1.xlsx')