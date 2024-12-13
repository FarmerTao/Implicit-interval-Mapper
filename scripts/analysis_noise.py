from gudhi.cover_complex import MapperComplex
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_all_Mapper(data, projected_data, n_comp, cl):
    g_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    G_list = []
    for g in g_list:
        mapper = MapperComplex(
        resolutions=n_comp,
        gains=g,
        clustering=cl,
        )
        mapper.fit(data, filters=projected_data.numpy(),colors=projected_data.numpy())
        G = mapper.get_networkx()
        G_list.append(G)
    

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # figsize可以根据需要调整
    for i in range(3):
        for j in range(3):
            # 获取当前图
            G = G_list[i * 3 + j]
            
            # 定义节点的位置，这里使用spring布局
            pos = nx.spring_layout(G)
            
            # 绘制图形
            nx.draw(G, pos, ax=axs[i, j], with_labels=True, node_color='lightblue', edge_color='gray')
            axs[i, j].set_title(f'{g_list[i * 3 + j]}')
    plt.show()

def plot_intervals(mode_assignments, projected_data):
    n_points, n_interval = mode_assignments.size()
    mode_assignments = mode_assignments.numpy()
    projected_data = projected_data.numpy()
    se_list = []

    for i in range(n_interval):
        col_index = i  
        non_zero_indices = np.nonzero(mode_assignments[:, col_index])[0]
        d = projected_data[non_zero_indices]
        #print(d)
        s,e = np.min(d), np.max(d)
        se_list.append((s,e))
        print(s,e)

    se_list = sorted(se_list, key=lambda point: point[0])
    for i,se in enumerate(se_list):
        s,e = se[0], se[1]
        plt.plot([s,e],[0.1*(i % 2)-1.6,0.1*(i % 2)-1.6])

    plt.show()
    return se_list

def get_intervals(mode_assignments, projected_data):

    n_points, n_interval = mode_assignments.shape
    mode_assignments = mode_assignments.numpy()
    projected_data = projected_data.numpy()
    se_list = []

    for i in range(n_interval):
        col_index = i  
        non_zero_indices = np.nonzero(mode_assignments[:, col_index])[0]
        d = projected_data[non_zero_indices]
        #print(d)
        s,e = np.min(d), np.max(d)
        se_list.append((s,e))
        #print(s,e)

    se_list = sorted(se_list, key=lambda point: point[0])

    return se_list


def get_regular_intervals(bounds, n_cubes, perc_overlap):
    ranges = bounds[1] - bounds[0]

    # |range| / (2n ( 1 - p))
    radius = ranges / (2 * (n_cubes) * (1 - perc_overlap) + 2*perc_overlap)
    print(radius)
    # centers are fixed w.r.t perc_overlap
    se_list = []
    s = bounds[0]
    for i in range(n_cubes):
        e = s + 2*radius
        se_list.append((s,e))
        s = s + (1-perc_overlap)*radius*2
    
    return se_list