import torch
import torch.nn as nn
import torch.distributions as D
from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
from itertools import combinations

def Lor_distance(X, c): #X: n*dim c: 1*dim
    # centering
    X = X-c
    # add x_0
    x0 =  torch.sqrt(torch.norm(X, p=2, dim=1)**2 + 1)
    X = torch.cat((x0.unsqueeze(1), X), dim=-1)
    g = torch.eye(X.size()[1],dtype=float)
    g[0,0] = -1
    term1 = torch.matmul(X,g)
    r = torch.matmul(term1, X.T)
    dis = torch.acosh(-r)
    dis.fill_diagonal_(fill_value=0)
    return dis

def get_extended_persistence_generators(simplextree):
    st=gd.SimplexTree(simplextree)
    filtration=[v[1] for v in st.get_skeleton(0)]
    dummy=len(filtration)
    dim=st.dimension()
    result=([[] for i in range(dim+1)],[])

    ## extend filtration
    st.extend_filtration()
    st.compute_persistence()
    
    ## get persistence pairs
    for pair in st.persistence_pairs():
        if len(pair[1]):

            pair_dim=len(pair[0])-1

            d1=dummy in pair[0]
            d2=dummy in pair[1]

            if d1: pair[0].remove(dummy)
            if d2: pair[1].remove(dummy)

            birth_list=[(-2*d1+1)*filtration[v] for v in pair[0]]
            birth=pair[0][birth_list.index(max(birth_list))]

            death_list=[(-2*d2+1)*filtration[v] for v in pair[1]]
            death=pair[1][death_list.index(max(death_list))]

            result[0][pair_dim].append([birth,death])

    for i in range(dim+1):
        result[0][i]=np.array(result[0][i])
    
    return(result)


# The parameters of the model are the vertex function values of the simplex tree.
def _ExtendedSimplexTree(simplextree, filtration, dimensions):
    # Parameters: simplextree (simplex tree on which to compute persistence)
    #             filtration (function values on the vertices of st),
    #             dimensions (homology dimensions),
    #             homology_coeff_field (homology field coefficient)
    
    simplextree.reset_filtration(-np.inf, 0)

    # Assign new filtration values
    for i in range(simplextree.num_vertices()):
        simplextree.assign_filtration([i], filtration[i])
    simplextree.make_filtration_non_decreasing()

    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = get_extended_persistence_generators(simplextree)
    
    L_indices = []
    for dimension in dimensions:
    
        finite_pairs = pairs[0][dimension] if len(pairs[0]) >= dimension+1 else np.empty(shape=[0,2])
        essential_pairs = pairs[1][dimension] if len(pairs[1]) >= dimension+1 else np.empty(shape=[0,1])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices

class ExtendedSimplexTreeLayer(nn.Module):
    def __init__(self, simplextree, homology_dimensions, min_persistence=None, homology_coeff_field=11, persistence_dim_max=False):
        super(ExtendedSimplexTreeLayer, self).__init__()
        self.simplextree = simplextree
        self.homology_dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(homology_dimensions)
        self.homology_coeff_field = homology_coeff_field
        self.persistence_dim_max = persistence_dim_max
        # Assert to check the length of min_persistence is equal to the length of homology_dimensions
        assert len(self.min_persistence) == len(self.homology_dimensions)

    def forward(self, filtration):
        # Compute the vertex pairs for which to compute persistence
        indices = _ExtendedSimplexTree(self.simplextree, filtration.data.numpy(), self.homology_dimensions)
        # Initialize list to store persistence diagrams
        self.dgms = []
        for idx_dim, dimension in enumerate(self.homology_dimensions):
            finite_dgm = torch.gather(filtration, 0, torch.tensor(indices[idx_dim][0], dtype=torch.long)).view(-1, 2)
            essential_dgm = torch.gather(filtration, 0, torch.tensor(indices[idx_dim][1], dtype=torch.long)).view(-1, 1)
            min_pers = self.min_persistence[idx_dim]
            if min_pers >= 0:
                # Filter finite persistence points based on minimum persistence threshold
                persistent_indices = (torch.abs(finite_dgm[:, 1] - finite_dgm[:, 0]) > min_pers).nonzero(as_tuple=False).squeeze()
                self.dgms.append((finite_dgm[persistent_indices], essential_dgm))
            else:
                self.dgms.append((finite_dgm, essential_dgm))
        
        return self.dgms

class Opt_GMM_Mapper(nn.Module):
    def __init__(self, n_components, weights = None, means = None, covariances = None, type = "point cloud"):
        super().__init__()

        self.n_components = n_components
        
        if weights == None:
            weights = torch.ones(self.n_components,requires_grad=True) / self.n_components
        self.weights = weights
        self.free_para = nn.Parameter(weights)
        
        if means is None:
            means = torch.randn(n_components, requires_grad= True)
        self.means = nn.Parameter(means)

        if covariances is None:
            covariances =  torch.ones(n_components, requires_grad= True)
        self.covariances = covariances
        self.var_para = nn.Parameter(torch.log(torch.sqrt(covariances)))

        self.log_likelihood = None

        self.Q = None
        self.type = type

        self.neg_log_likelihood = None
        self.var_regularization = None

        self.dgm_list = None
        self.topo_loss = None
        self.mode_dgm = None
        self.topo_mode_loss = None

    def forward(self, f, X, clustering): #filtered data, data
        self.f = f
        self.data = X
        self.clustering = clustering

        # update weights and variance by using free parameters
        self.weights = torch.exp(self.free_para)/torch.exp(self.free_para).sum()
        nan_mask = torch.isnan(self.weights)
        self.weights[nan_mask] = 1/self.n_components
        self.covariances = torch.exp(self.var_para)**2

        #define GMM model
        gmm = GMM(self.weights, self.means, self.covariances)

        #compute Q for mutilnal distribution parameters
        log_probs = gmm.pdf(f).log()
        self.log_likelihood = log_probs

        # check NaN
        Q = gmm.component_probs(f)
        Q = Q.T  
        # subtitute with a very small number
        nan_mask = torch.isnan(Q)
        zeros_tensor = torch.zeros_like(Q)
        Q = torch.where(nan_mask, zeros_tensor + 1e-12, Q)
        Q = torch.clamp(Q, min=1e-12, max=1-1e-12)
        self.Q = Q

        # threotical of mode_assignments
        mode_assignments = self._row_mode(Q)
        self.mode_assignments = mode_assignments


        #check
        rows_all_zeros = mode_assignments.sum(dim=1) == 0
        indices_all_zeros = torch.nonzero(rows_all_zeros, as_tuple=True)
        assert indices_all_zeros[0].sum() == 0., print("Indices of rows that are all zeros:", indices_all_zeros)

        mode_assignments = mode_assignments.unsqueeze(0) # 1x (num_points) x (resolution)
        
        mode_clusters = compute_mapper(X, clustering, mode_assignments, input_type = self.type)
        mode_clusters = torch.Tensor(mode_clusters) #{0,1}
        mode_clusters = mode_clusters.to(X.dtype) # 1 x (num_points) x (resolution)

        # compute filtration for Mapper node
        mode_filtration = compute_filtration_ll(mode_clusters,self.log_likelihood) # 1 x (num_points) x (resolution)
        mode_filtration_f = compute_filtration(mode_clusters,f)
        # delete all 0 column
        mode_clusters = mode_clusters[0]
        mode_filtration = mode_filtration[0]
        zero_indice = torch.all(mode_clusters == 0, dim=0)
        mode_clusters = mode_clusters[:, ~zero_indice]
        mode_filtration = mode_filtration[~zero_indice] #nan去掉

        self.mode_filtration = mode_filtration
        self.mode_filtration_f = mode_filtration_f[0][~zero_indice]
        self.mode_clusters = mode_clusters

        # construct simplex tree
        simplextree = gd.SimplexTree()

        # add edges
        indices_combination = list(combinations(range(mode_clusters.shape[1]), 2))

        for combine in indices_combination:
            temp = torch.dot(mode_clusters[:,combine[0]],mode_clusters[:,combine[1]])
            if temp != 0:
                simplextree.insert(list(combine))

        exist_v = []
        for (v,_) in simplextree.get_skeleton(0):
            exist_v.append(v[0])

        # add nodes 
        for j in range(mode_filtration.shape[0]):
            if j not in exist_v:
                simplextree.insert([j])

        layer = ExtendedSimplexTreeLayer(simplextree, homology_dimensions=[0,1])
        
        #compute dgm
        self.mode_dgm = layer(mode_filtration)
        
        topo_mode_loss = 0.
        num_points = 0 # num of points in dgm
        for d in self.mode_dgm:   
            l = len(d[0].size())
            if torch.numel(d[0]) != 0:
                if l == 1:
                    num = 1 
                    topo_mode_loss = topo_mode_loss + (d[0][0] - d[0][1]).abs().sum()
                if l > 1:
                    num = d[0].size(0) 
                    topo_mode_loss = topo_mode_loss + (d[0][:, 0] - d[0][:, 1]).abs().sum()
            else:
                num = 0
                topo_mode_loss = topo_mode_loss + d[0].sum() + 1e-12
            num_points = num_points + num
        #print(num_points)
        self.topo_mode_loss = - topo_mode_loss / num_points # average
        #print(self.topo_mode_loss)
        return Q

    def _row_mode(self,scheme): # (num_points) x (resolution)
        scheme = scheme.T
        assignments = torch.zeros(scheme.shape[0],scheme.shape[1])
        for (i,s) in enumerate(scheme):
            top2_elements, top2_indices = torch.topk(s, 2, largest=True, sorted=True)
            if 0.5*top2_elements[0] > top2_elements[1]:
                assignments[i, top2_indices[0]] = 2
            else:
                assignments[i, top2_indices[0]] = 1
                assignments[i, top2_indices[1]] = 1

        return assignments

    
    def loss(self,l1,l2):
        self.neg_log_likelihood = - self.log_likelihood.sum() / len(self.log_likelihood) #average loglikelihood
        # print(self.neg_log_likelihood)
        loss = l1*self.neg_log_likelihood + l2*self.topo_mode_loss 
        return loss
    
    def draw(self):
        x = torch.linspace(float(torch.min(self.means))-5,float(torch.max(self.means))+5, 1000)
        mix = D.Categorical(probs=self.weights)
        comp = D.Normal(self.means, torch.sqrt(self.covariances)) 
        gmm = D.MixtureSameFamily(mix,comp)

        # draw pdf
        pdf = gmm.log_prob(x).exp()

        plt.plot(x.detach().numpy(), pdf.detach().numpy())
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title('GMM Probability Density Function')
        plt.show()

    def get_mode_graph(self):
        G = generate_graph(self.mode_clusters,self.mode_filtration)
        return G
    
    def sample(self,K, scheme):
        f = self.f
        f = f.repeat(1, K).T
        f = torch.unsqueeze(f, dim=2)

        upscheme = scheme.repeat(K,1,1)
        
        # sample H from Multinomial distribution
        assignments = D.Multinomial(total_count=2, probs=upscheme.permute(0, 2, 1)).sample()
  
        self.assignments = assignments

        # Mapper function
        clusters = compute_mapper(self.data, self.clustering, assignments,input_type=self.type)

        self.sample_clusters = clusters

        # compute filter for each node
        filtration = compute_filtration(clusters,f)
        self.sample_filtration = filtration


        G_list = []

        for k in range(K):
            clusters_k = clusters[k]
            filtration_k = filtration[k]
            G = generate_graph(clusters_k,filtration_k)
            G_list.append(G)

        return G_list #return networkx graph
