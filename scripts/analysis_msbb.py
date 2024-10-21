import torch
import numpy as np
import pandas as pd
import math
import graph_tool as gt
from graph_tool.draw import graph_draw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chisquare
import torch.distributions as D

def query_braak(data_num, data_path, verbo = False):
    data = pd.read_csv(data_path,index_col=0)
    id_list = data.index
    id = id_list[data_num]

    indi_speci_data = pd.read_csv("data/MSBB_biospecimen_metadata.csv")
    indi_speci_data = np.array(indi_speci_data)
    indi_speci_data = indi_speci_data[:,[0,1]]

    id_loc = np.where(indi_speci_data == id)
    individualID = indi_speci_data[id_loc[0],id_loc[1]-1]
    if len(individualID) == 0:
        return 0
    individualID = individualID[0]
    if type(individualID) != str:
        if math.isnan(individualID):
            return 0

    #17
    indi_braak_data = pd.read_csv("data/MSBB_individual_metadata.csv")
    indi_braak_data = np.array(indi_braak_data)
    b_loc = np.where(indi_braak_data == individualID)
    braak = indi_braak_data[b_loc[0],17][0]
    if verbo:
        print("individualID:{}".format(individualID))
        print("braak:{}".format(braak))

    return braak

def qurey_node_id(clusters, node_j):
    non_zero_indices = torch.nonzero(clusters[:,node_j])
    non_zero_indices = non_zero_indices[:,0].tolist()
    non_zero_indices = list(set(non_zero_indices))
    return non_zero_indices

def qurey_frac(clusters,node_j,data_path):
    non_zero_indices = torch.nonzero(clusters[:,node_j])
    non_zero_indices = non_zero_indices[:,0].tolist()
    non_zero_indices = list(set(non_zero_indices))
    id_list = [] 
    for ind in non_zero_indices:
        id_list.append(query_braak(int(ind),data_path))
    
    count  = list(np.bincount(np.array(id_list)))
    
    l = len(count)
    if l < 7 :
        count = count + [0]*(7-l)
    count = np.array(count)
    return count/count.sum(), count


def qurey_frac_all(clusters,data_path):
    braak_list = [] # braak list for all inds
    for ind in range(clusters.size()[0]):
        braak_list.append(query_braak(int(ind),data_path))
    
    count  = list(np.bincount(np.array(braak_list)))
    l = len(count)
    if l < 7 :
        count = count + [0]*(7-l)
    count = np.array(count)
    print(count.sum())
    return count/count.sum(), count

def analysis_braak(G_mode, train, data_path, projected_data, good_nodes, x_lim, y_lim,
                    pos_list = None, rect_pos = None, text_pos = None,name = "example" ):
    clusters = train.mode_clusters
    clusters = clusters.squeeze(dim = 0)
    clusters = clusters[:, ~torch.all(clusters == 0, dim=0)] # 0,1 matrix

    #color_list = ['#F5F5F5','#DCDCDC','#C0C0C0','#969696','#656565','#404040','#000000']
    #color_list = ['#54B345','#32B897','#BB9727','#05B9E2','#8983BF','#C76DA2','#F27970']
    color_list = ["#76da91", "#63b2ee", "#7cd6cf","#9192ab", "#f8cb7f", "#efa666", "#f89588"]
    graph_gt = gt.Graph(directed=False)
    pie_fractions = graph_gt.new_vertex_property("vector<double>")
    pie_colors = graph_gt.new_vertex_property("vector<string>")
    node_text = graph_gt.new_vertex_property("string")

    vertex_map = {node: graph_gt.add_vertex() for node in G_mode.nodes()}

    # add
    for i in G_mode.nodes():
        frac, count = qurey_frac(clusters,i,data_path)
        pie_fractions[i] = list(frac)
        pie_colors[i] = color_list
        node_text[i] = "{}".format(int(count.sum()))
        #node_text[i] = "{}".format(i)  #node id

    for i,e in enumerate(G_mode.edges()):
        u,v = e
        graph_gt.add_edge(vertex_map[u], vertex_map[v])

    
    vertex_index = graph_gt.vertex_index
    pos = gt.draw.sfdp_layout(graph_gt)

    if pos_list != None:
        for v in graph_gt.vertices():
            ind = vertex_index[v]
            pos[v] = pos_list[ind]
    else:
        pos_list = []
        for v in graph_gt.vertices():
            pos_list.append(pos[v])
            print("Vertex", v, "position:", list(pos[v]))
        print(pos_list)
    
    graph_draw(graph_gt, pos = pos, vprops={"shape": "pie", "pie_fractions": pie_fractions, "pie_colors": pie_colors,
                                "size":40,
                                "text_position":1, "font_size":20,"text": node_text,
                                "text_offset":text_pos,"text_rotation":0.},output="graph.eps")

    #text_position":2, "font_size":20,"text": node_text, 
    plt.figure(figsize=(12, 9),dpi=300)  
    plt.subplot(221)
    plt.imshow(plt.imread('graph.eps'), aspect='auto')  
    plt.axis('off')  
    plt.title("(a)")

    # 画方框
    if isinstance(rect_pos, tuple) :
        left, bottom, width, height = rect_pos
        rect=mpatches.Rectangle((left,bottom),width,height, 
                                fill=False,
                                color="red",
                            linewidth=2,
                            linestyle='--')
                            #facecolor="red")
        plt.gca().add_patch(rect)

    if isinstance(rect_pos, list):
        #print("list list!!!!")
        for rec in rect_pos:
            left, bottom, width, height = rec
            rect=mpatches.Rectangle((left,bottom),width,height, 
                                    fill=False,
                                    color="red",
                                linewidth=2,
                                linestyle='--')
                                #facecolor="red")
            plt.gca().add_patch(rect)

    frac_all, count_all = qurey_frac_all(clusters,data_path)
    labels = ['0', '1','2','3','4','5','6']
    radius = 0.01  # 一个很小的半径
    proxy_wedges = []
    for i, color in enumerate(color_list):
        # 创建一个圆形楔形作为代理艺术家
        wedge = mpatches.Wedge(center=(0, 0), r=radius, theta1=0, theta2=360, color=color)
        proxy_wedges.append(wedge)
    plt.legend(handles=proxy_wedges, labels=labels, fontsize='small')
    #ax[1].pie(frac_all, labels=['0', '1','2','3','4','5','6'], colors=color_list)

    _, count = qurey_frac(clusters,good_nodes,data_path)
  

    # 对应的类别
    categories = [0,1,2,3,4,5,6]
    plt.subplot(222)
    # 创建柱状图
    plt.bar(categories, count, alpha = 0.5, color = "#FEA3A2", label = "Nodes inside the red dashed box")
    plt.bar(categories, count_all, bottom=count, color = "#8E8BFE", label = "All nodes")
    # 添加标题和标签
    #plt.bar_label(b)
    plt.legend()
    plt.title('(b)')
    plt.xlabel('Braak value')
    plt.ylabel('number of sample')
    plt.legend(fontsize='small')

    # GMM PDF
    m = train.Mapper
    x = torch.linspace(float(torch.min(m.means))-0.1,float(torch.max(m.means))+0.1, 1000)
    mix = D.Categorical(probs=m.weights)
    comp = D.Normal(m.means, torch.sqrt(m.covariances.abs())) 
    gmm = D.MixtureSameFamily(mix,comp)
    pdf = gmm.log_prob(x).exp()

    plt.subplot(212)
    # density
    plt.plot(x.detach().numpy(), pdf.detach().numpy())

    mode_H = train.mode_assignmnets
    #mode_H = mode_H[:, ~torch.all(mode_H == 0, dim=0)] 


    labels = []
    num_points,num_classes = mode_H.size()
    for i in range(num_points):
        for j in range(num_classes):
            if mode_H[i][j] >= 1.:
                labels.append(j)
                break
    labels = torch.tensor(labels)

    # (color, mark)

    class_colors = ['#FFBE7A','#FA7F6F','#82B0D2']
    marks = [ ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
    "h", "H", "+", "x", "X", "D", "d", "|", "_" ]

    color_mark_bank = []
    color_index = 0
    mark_index = 0
    for j in range(num_classes):
        color_index += 1
        mark_index += 1

        if color_index == len(class_colors):
            color_index = 0

        if mark_index == len(marks):
            mark_index = 0
        color_mark_bank.append([class_colors[color_index], marks[mark_index]])


    # 重新排一下序
    means, indices = torch.sort(m.means)
    mode_H = mode_H[:,indices]

    #check, delete the class that has no points
    check = mode_H.sum(dim = 0)
    zero_indices = torch.nonzero(check == 0)
    if zero_indices.numel() != 0:
        print("class{} has no points!!!".format(list(zero_indices)[0]))

    braak_list = []
    for i in range(len(projected_data)):
        braak_list.append(query_braak(i,data_path))
        
    for j in range(num_classes):
        non_zero_indices = torch.nonzero(mode_H[:,j])
        non_zero_indices = non_zero_indices.squeeze(dim=-1)
        points = projected_data[non_zero_indices]
        color_mark = color_mark_bank[j]
        plt.scatter(points, [braak_list[k]-7 for k in non_zero_indices.tolist()], s=20,
                    c = color_mark[0], marker = color_mark[1], label=f'Class {j}')

    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('(c)')
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    ax2 = plt.twinx()
    
    #ax2.scatter(points, [braak_list[k] for k in non_zero_indices.tolist()], color='white')
    ax2.set_ylabel('Braak value')
    ax2.set_ylim((-1,y_lim[1]+7))
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6])
    plt.tight_layout()
    plt.savefig("{}.eps".format(name))
    plt.show()

    print("###########")
    #test_result = ks_2samp(count, count_all)
    count_all = count_all - count
    test_result = chisquare(f_obs=count, f_exp=count_all/sum(count_all)*sum(count))
    print(test_result)
    pvalue = test_result.pvalue
    if pvalue < 0.05:
        print("Have difference!!!")
    elif pvalue >= 0.05:
        print("No differenece!!!")

    return