import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 导入数据
data = pd.read_excel('data6.xlsx', header=None)
data = data.values[1:]
G = nx.Graph()

def draw():
    # 得到坐标
    vnode = data[...,1:3]
    # 得到点并导入
    nodes = list(data[...,0])
    G.add_nodes_from(nodes)
    # 导入边
    edges = []
    for l in data:
        n1 = l[0]
        for n2 in l[4:]:
            if n2 is not np.nan:
                edges.append([n1,n2])
    G.add_edges_from(edges)
    npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示
    # 导入标志
    # 绘制五角星
    # ax = plt.figure()
    labels = []
    for l in data[...,3]:
        if l==1:
            labels.append('★')
        elif l==2:
            labels.append('*')
        else:
            labels.append('·')
    nlabels = dict(zip(nodes, labels))  # 标志字典，构建节点与标识点之间的关系
    nx.draw_networkx_nodes(G, npos, node_size=50, node_color="#6CB6FF")  # 绘制节点
    nx.draw_networkx_edges(G, npos, edges)  # 绘制边
    nx.draw_networkx_labels(G, npos, nlabels)  # 标签
    # nx.draw_networkx(G)
    plt.show()

def distant(vnode, i, j):
    return np.sqrt(np.square(vnode[i,0]-vnode[j,0]) + np.square(vnode[i,1]-vnode[j,1]))

# def minTree():
def minTree():
    # 得到坐标
    vnode = data[...,1:3]
    # 得到点并导入
    nodes = list(data[...,0])
    # 获取节点与坐标之间的映射关系，用字典表示
    npos = dict(zip(nodes, vnode))
    # 获得最小生成树
    T = nx.minimum_spanning_tree(G)
    # 得到邻接矩阵
    C = nx.to_numpy_matrix(T)
    # 计算两点距离
    for i in range(95):
        for j in range(95):
            if C[i,j] == 1:
                C[i,j] = distant(vnode, i, j)
    w = C.sum()/2
    print("最先生成树的权重 W=\n", w)
    # 画图
    nx.draw(T, npos, with_labels=True, node_size=50, node_color="#6CB6FF")
    w2 = nx.get_edge_attributes(T, 'weight')
    nx.draw_networkx_edge_labels(T, npos, edge_labels=w2)
    plt.show()

def path():
    # 得到坐标
    vnode = data[...,1:3]
    # 得到点并导入
    nodes = list(data[...,0])
    # 节点名映射到 int ，方便计算距离
    nodes2int = {}
    for i in range(len(nodes)):
        nodes2int[nodes[i]] = i
    # 导入点
    G.add_nodes_from(nodes)
    # 导入边
    edges = []
    for l in data:
        n1 = l[0]
        for n2 in l[4:]:
            if n2 is not np.nan:
                edges.append([n1,n2,distant(vnode, nodes2int[n1], nodes2int[n2])])
    G.add_weighted_edges_from(edges)
    # 映射坐标
    npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示
    # 计算最短路径
    print('dijkstra 方法寻找最短路径：')
    path = nx.dijkstra_path(G, source='L', target='R3')
    print('节点 L 到 R3 的路径：', path)
    print('dijkstra 方法寻找最短距离：')
    distance = nx.dijkstra_path_length(G, source='L', target='R3')
    print('节点 L 到 R3 的距离为：', distance)
    # 画出最短路径
    labels = []
    for l in data[...,0]:
        if l in path:
            labels.append('★')
        else:
            labels.append('·')
    nlabels = dict(zip(nodes, labels))  # 标志字典，构建节点与标识点之间的关系
    nx.draw_networkx_nodes(G, npos, node_size=50, node_color="#6CB6FF")  # 绘制节点
    nx.draw_networkx_edges(G, npos, edges)  # 绘制边
    nx.draw_networkx_labels(G, npos, nlabels)  # 标签
    plt.show()

if __name__ == '__main__':
    draw()
    minTree()
    path()
