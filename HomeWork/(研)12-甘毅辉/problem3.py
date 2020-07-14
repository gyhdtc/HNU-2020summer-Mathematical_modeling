from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
borrow_money = 5000
# students_money = 200

np.set_printoptions(precision=3)
risk = [5,5,5,2,2,2,4,4,4,3,3,3,1,1,1]
taxrate = [0,0,0,0.2,0.2,0.2,0,0,0,0.1,0.1,0.1,0.3,0.3,0.3]
deadline = [2,9,20,3,12,25,4,15,20,4,9,18,2,5,18]
profitrate = [4.45,39.89,232.11,8.0,58.27,300.54,12.55,180.09,232.11,12.55,39.89,206.12,4.45,18.77,206.12]
# 优化目标函数

def func(x):
    x = np.reshape(x,[16,28])
    x = x[:-1]
    Profit = np.zeros([15,30])
    for i in range(Profit.shape[0]):
        for j in range(Profit.shape[1]):
            if j+ deadline[i]<30:

                Profit[i][j+ deadline[i]] = x[i][j]*(0.01*profitrate[i])*(1-taxrate[i])
    area_invest = np.sum(x, axis=1)  # 15
    industry_invest = np.sum(area_invest.reshape(5, 3), axis=1)  # 5
    return -np.sum(Profit)+10*((industry_invest[0]-3*industry_invest[1])**2+(industry_invest[0]-industry_invest[2])**2+(industry_invest[0]-2*industry_invest[3])**2+(industry_invest[0]-4*industry_invest[4])**2)

# 投资比例约束1
def conv1_1(x):
    x = np.reshape(x,[16,28])
    x = x[:-1]
    area_invest = np.sum(x,axis=1) # 15
    industry_invest = np.sum(area_invest.reshape(5,3),axis=1) # 5
    return industry_invest[0]-3*industry_invest[1]

# 投资比例约束2
def conv1_2(x):
    x = np.reshape(x,[16,28])
    x = x[:-1]
    area_invest = np.sum(x,axis=1) # 15
    industry_invest = np.sum(area_invest.reshape(5,3),axis=1) # 5
    return industry_invest[0]-1*industry_invest[1]
# 投资比例约束3
def conv1_3(x):
    x = np.reshape(x,[16,28])
    x = x[:-1]
    area_invest = np.sum(x,axis=1) # 15
    industry_invest = np.sum(area_invest.reshape(5,3),axis=1) # 5
    return industry_invest[0]-2*industry_invest[1]
# 投资比例约束4
def conv1_4(x):
    x = np.reshape(x,[16,28])
    x = x[:-1]
    area_invest = np.sum(x,axis=1) # 15
    industry_invest = np.sum(area_invest.reshape(5,3),axis=1) # 5
    return industry_invest[0]-4*industry_invest[1]

# 投资量约束
def conv2(x):
    x = np.reshape(x, [16, 28])
    x = x[:-1]

    store_money = x[-1]

    Profit = np.zeros([15, 30])
    adprofit = np.zeros([15, 30])
    for i in range(Profit.shape[0]):
        for j in range(Profit.shape[1]):
            if j + deadline[i] < 30:
                Profit[i][j+ deadline[i]] = x[i][j] * (0.01 * profitrate[i]) * (1 - taxrate[i])
                adprofit[i][j+ deadline[i]] = x[i][j] * (1+0.01 * profitrate[i]) * (1 - taxrate[i])
    profitsum = np.sum(Profit)
    students_money = profitsum/30
    base_money = np.zeros([28])
    # 每年的本金
    base_money[0] = 8000-students_money+borrow_money
    base_money[1] = store_money[0]-store_money[1]-students_money-borrow_money*0.02
    for i in range(1,base_money.shape[0]-1):
        base_money[i+1] = store_money[i]-students_money+np.sum(adprofit[...,i])-store_money[i+1]
        # base_money[i + 1] = np.sum(adprofit[..., i])
        # print(base_money[i])
    # 每年的投资
    year_invest = np.sum(x,axis=0) # 30
    return base_money - year_invest

# 年限约束
def conv3(x):
    x = np.reshape(x, [16, 28])
    x = x[:-1]
    area_invest = np.sum(x, axis=1) # 15
    date = 0
    for i in range(len(deadline)):

        date+= deadline[i]*area_invest[i]
    # 平均年限
    avgdata = date/np.sum(x)
    return 8-avgdata

# 风险约束
def conv4(x):
    x = np.reshape(x, [16, 28])
    x = x[:-1]
    area_invest = np.sum(x, axis=1) # 15
    risksum = 0
    for i in range(len(risk)):
        risksum+= risk[i]*area_invest[i]
    # 平均风险
    avgrisk = risksum/np.sum(x)
    return avgrisk-2.5



def f():
# global students_money
# for i in range(10):
#     students_money=students_money-i*10

    con11 ={'type': 'eq', 'fun': conv1_1}
    con12 ={'type': 'eq', 'fun': conv1_2}
    con13 ={'type': 'eq', 'fun': conv1_3}
    con14 ={'type': 'eq', 'fun': conv1_4}
    con2 ={'type': 'ineq', 'fun': conv2}
    con3 ={'type': 'ineq', 'fun': conv3}
    con4 ={'type': 'ineq', 'fun': conv4}

    con = [con2,con3,con4]
    b = (0,8000)
    bd = [b for i in range(448)]
    res = minimize(func, np.zeros([448])+10+1e-5, options={'maxiter':100000,'disp':True},constraints=con,bounds=bd)
    x = res['x']
    print(np.round(np.reshape(x,[16,28]),2))
    # print(store_money)

    print('max profit',func(x))
    print(conv1_1(x))
    print(conv1_2(x))
    print(conv1_3(x))
    print(conv1_4(x))
    print('本金-投资')
    print(conv2(x))
    print(conv3(x))
    print(np.round(conv4(x),2))
    print(res['message'])
    x = np.reshape(x,[16,28])
    x = x[:-1]
    a2=pd.DataFrame(x)
    a2.to_excel('data1.xlsx', index=False)  #不包括行索引

    Profit = np.zeros([15,30])
    for i in range(Profit.shape[0]):
        for j in range(Profit.shape[1]):
            if j+ deadline[i]<30:

                Profit[i][j+ deadline[i]] = x[i][j]*(0.01*profitrate[i])*(1-taxrate[i])
    area_invest = np.sum(x, axis=1)  # 15
    industry_invest = np.sum(area_invest.reshape(5, 3), axis=1)  # 5
    print(-np.sum(Profit))
    print(industry_invest[0])
    print(industry_invest[1]*3)
    print(industry_invest[2])
    print(industry_invest[3]*2)
    print(industry_invest[4]*4)
if __name__ == '__main__':
    f()
