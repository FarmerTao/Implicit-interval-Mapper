import numpy as np
import torch
from itertools import combinations
import gudhi as gd
import matplotlib.pyplot as plt
import networkx as nx

def cluster_patch(Binned_data,k,p_ind,X,clustering,maximum,input_type):
    """
    Clusters a patch. Used to compute multiple Mappers in parallel.

    Parameters:
        Binned_data (list of length num_mappers containing lists of length resolution): list containing the indexes of data points located in each patch for each Mapper
        k (integer): index of Mapper
        p_ind (integer): index of patch
        X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
        clustering (class): clustering class. Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
        maximum (integer): maximum number of clusters possible in one patch.
        input_type (string): type of input data. Either "point cloud" or "distance matrix".
    
    Returns:
        clusters (numpy array of shape (n_samples,)): Index of the cluster each point in the patch belongs to.
        k (integer): index of Mapper
        p_ind (integer): index of patch
    """
    data_bin=Binned_data[k][p_ind] #index of data

    if input_type == "point cloud": 
        if len(data_bin) > 1:
            clusters = clustering.fit_predict(X[data_bin,:]) 
        elif len(data_bin) == 1:
            clusters = np.array([0])
        else:
            clusters = np.array([])

    if input_type == "distance matrix": 
        if len(data_bin) > 1:
            clusters = clustering.fit_predict(X[data_bin,:][:,data_bin])
        elif len(data_bin) == 1:
            clusters = np.array([0])
        else:
            clusters = np.array([])

    clusters=clusters+p_ind*maximum
    return clusters,k,p_ind

def compute_mapper(X,clustering,assignments,input_type="point cloud",maximum=10):
    """
    Computes Mappers belonging to several cover assignments in Parallel. Only supports one dimensional filters.

    Parameters:
        X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
        clustering (class): clustering class. Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
        assignments (numpy array of shape (num_mappers) x (resolution) x (num_points) containing values in {0,1}): Cover assignments for each Mapper.
        input_type (string): type of input data. Either "point cloud" or "distance matrix".
        maximum (integer): maximum number of clusters possible in one patch.
    
    Returns:
        simplex_trees (list of gd.SimplexTree() instances of length num_mappers): list of Mapper simplicial complexes represented by gudhi Simplex Trees. For computational efficiency considerations, empty clusters are present as isolated vertices in the simplicial complexes.
        clusters_array (numpy array of shape (num_mappers) x (num_points) x (maximum x resolution) containing values in {0,1}): Belongings, encoded as ones and zeros, of the points to the clusters in each Mapper.
    """
    assignments = assignments.permute(0, 2, 1) #(num_mappers) x (resolution) x (num_points)
    K,resolution,num_pts=assignments.shape

    clusters_array =  torch.zeros((K,num_pts,maximum*resolution),requires_grad=False)
    # Compute the Binned data map that associates each patch to the points that belong to it
    Binned_data=[[torch.where(assignments[k,p,:]>0)[0] for p in range(resolution)] 
                 for k in range(K)] #index of data in each bin
    
    # Compute the clustering in each patch in parallel

    parameter_list = [(Binned_data, k, p, X, clustering, maximum, input_type) for k in range(K) for p in range(resolution)]
    for params in parameter_list:
        result = cluster_patch(*params)

        #clusters_list.append(result) 
        if result[0].size!=0:
            clusters_array[result[1],Binned_data[result[1]][result[2]],result[0]]=1 
    return clusters_array

def compute_dgm(clusters_k,filtration_k): 
        

    clusters_k = clusters_k[:, ~torch.all(clusters_k == 0, dim=0)]
    filtration_k = filtration_k[~torch.isnan(filtration_k)] #

    simplextree = gd.SimplexTree()

    # add edges
    indices_combination = list(combinations(range(clusters_k.shape[1]), 2))

    for combine in indices_combination:
        temp = torch.dot(clusters_k[:,combine[0]],clusters_k[:,combine[1]])
        if temp != 0:
            simplextree.insert(list(combine))

    exist_v = []
    for (v,_) in simplextree.get_skeleton(0):
        exist_v.append(v[0])

    # add nodes
    for j in range(filtration_k.shape[0]):
        if j in exist_v:
            simplextree.assign_filtration([j], float(filtration_k[j]))
        else:
            simplextree.insert([j])
            simplextree.assign_filtration([j], float(filtration_k[j]))

    simplextree.make_filtration_non_decreasing()
    
    simplextree.extend_filtration()

    dgms = simplextree.extended_persistence() 

    dgm = []
    
    for d in dgms:
        if d != []:
            for p in d:
                dgm.append(p)
    
    return dgm

def draw_graph(clusters_k,filtration_k,path='figures',name = 'example',format = 'eps', save_fig = True):

    filtration_k = filtration_k[~torch.all(clusters_k == 0, dim=0)]
    clusters_k = clusters_k[:, ~torch.all(clusters_k == 0, dim=0)]
    #filtration_k = filtration_k[~torch.isnan(filtration_k)]
    
    # create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(len(filtration_k)))
    indices_combination = list(combinations(range(clusters_k.shape[1]), 2))
    
    for combine in indices_combination:
        
        temp = torch.dot(clusters_k[:,combine[0]],clusters_k[:,combine[1]])
        if temp != 0:
            G.add_edge(combine[0],combine[1])

    pos = nx.spring_layout(G)  
    nx.draw(G, pos=pos, with_labels=False, node_color = filtration_k.detach().numpy())  # 绘制图形 , node_color = filtration_k
    path = path + "/" + name + "." + format

    if save_fig:
        plt.savefig(path)
        
    plt.show()  

    return G

def generate_graph(clusters_k,filtration_k):

    filtration_k = filtration_k[~torch.all(clusters_k == 0, dim=0)]
    clusters_k = clusters_k[:, ~torch.all(clusters_k == 0, dim=0)]
    #filtration_k = filtration_k[~torch.isnan(filtration_k)]
    
    # create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(len(filtration_k)))
    indices_combination = list(combinations(range(clusters_k.shape[1]), 2))
    
    for combine in indices_combination:
        
        temp = torch.dot(clusters_k[:,combine[0]],clusters_k[:,combine[1]])
        if temp != 0:
            G.add_edge(combine[0],combine[1])

    return G

def compute_filtration(clusters,f): #clusters_array (numpy array of shape (num_mappers) x (num_points) x (maximum x resolution)
    clusters = clusters.to(f.dtype)
    
    sum_clusters = torch.sum(clusters, dim=1)  # 按行求和 (numpy array of shape (num_mappers) x (maximum x resolution) x 1 每一类有几个点
    sum_f_values_clusters = torch.matmul( clusters.transpose(1, 2),f)  

    sum_f_values_clusters = torch.squeeze(sum_f_values_clusters, dim=2)

    filtration = torch.div(sum_f_values_clusters, sum_clusters)  

    return filtration #filtration (numpy array of shape (num_mappers) x (maximum x resolution)  

def compute_filtration_ll(clusters,ll): #average ll for each node 
    clusters = clusters.to(ll.dtype)
    sum_clusters = torch.sum(clusters, dim=1)
    sum_clusters = torch.clamp(sum_clusters, min=1e-12)
    sum_f_values_clusters = torch.matmul( clusters.transpose(1, 2),ll)  

    sum_f_values_clusters = torch.squeeze(sum_f_values_clusters, dim=2)
    filtration = torch.div(sum_f_values_clusters, sum_clusters)
    
    return filtration #sum_f_values_clusters #(numpy array of shape (num_mappers) x (maximum x resolution)  

class GMM:
    def __init__(self, weights, means, covariances):
        """
        初始化一维高斯混合模型。
        :param weights: 每个高斯成分的权重，形状为 (n_components,)。
        :param means: 每个高斯成分的均值，形状为 (n_components,)。
        :param covariances: 每个高斯成分的方差，形状为 (n_components,)。
        """
        self.weights = weights
        self.means = means
        self.covariances = covariances

    def pdf(self, x):
        """
        计算GMM在给定数据点x上的pdf。
        :param x: 输入数据点，可以是单个数值或一个数组，形状为 (batch_size,)。
        :return: 数据点x在GMM模型下的pdf值。
        """
        batch_size = x.shape[0]
        pdf_values = torch.zeros(batch_size,1)
        
        for i in range(self.weights.shape[0]):
            weight = self.weights[i]
            mean = self.means[i]
            covariance = self.covariances[i]
            norm_term = 1 / torch.sqrt(2 * torch.pi * covariance)
            norm_term = norm_term.repeat(batch_size,1)
            exponent = -0.5 * ((x - mean) ** 2) / covariance

            pdf_values = pdf_values.clone().add(weight * torch.exp(exponent) * norm_term) 
        
        return pdf_values
    
    def component_probs(self, x):
        """
        Compute the assignment probability of each point
        :param x: input data points, shape: (batch_size,)。
        :return: compute probability of each point, shape: (batch_size, n_components)。
        """
        
        batch_size = x.shape[0]
        n_components = self.weights.shape[0]
        
        # initialize
        component_probs = torch.zeros(batch_size, n_components)

        for i in range(n_components):
            weight = self.weights[i]
            mean = self.means[i]
            covariance = self.covariances[i]
            norm_term = 1 / torch.sqrt(2 * torch.pi * covariance)
            norm_term = norm_term.repeat(batch_size,1)
            exponent = -0.5 * ((x - mean) ** 2) / covariance
            component_probs[:, i] = (weight * torch.exp(exponent) * norm_term).squeeze()
        
        component_probs = component_probs / component_probs.sum(dim=1, keepdim=True)
        return component_probs
    
    