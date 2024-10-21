import torch
import torch.distributions as D
from .utils import compute_mapper,draw_graph,compute_filtration

class GMM_Soft_Mapper():
    def __init__(self, scheme, clustering,
                 data, projected_data,
                 path='figures',name = 'example',format = 'eps',
                 type = "point cloud"):
        self.scheme = scheme # hidden assignment matrix H
        self.clustering = clustering
        self.type = type

        self.f = projected_data
        self.data = data

        # save path
        self.path = path
        self.name = name
        self.format = format

        # save state for every point in sample Mapper
        self.assignments = None
        self.clusters = None
        self.filtration = None

        # save state for every point in mode Mapper
        self.mode_assignments = None
        self.mode_clusters = None
        self.mode_filtration = None

    def sample(self,K,plot_num = None, save_fig = True):
        f = self.f
        f = f.repeat(1, K).T
        f = torch.unsqueeze(f, dim=2)

        upscheme = self.scheme.repeat(K,1,1)
        
        # sample H from Multinomial distribution
        assignments = D.Multinomial(total_count=2, probs=upscheme.permute(0, 2, 1)).sample()
  
        self.assignments = assignments

        # Mapper function
        clusters=compute_mapper(self.data, self.clustering, assignments,input_type=self.type)

        self.clusters = clusters

        # compute filter for each node
        filtration = compute_filtration(clusters,f)
        self.filtration = filtration

        # plot sample Mapper 
        if plot_num == None:
            plot_num = K

        G_list = []

        for k in range(plot_num):
            clusters_k = clusters[k]
            filtration_k = filtration[k]
            G = draw_graph(clusters_k,filtration_k,self.path,self.name+str(k),self.format,save_fig)
            G_list.append(G)

        return G_list #return networkx graph
    
    def compute_mode(self,save_fig = True): #compute mode for a set of samples

        print("mode:")
        
        def _row_mode(vectors): #(num_mappers) x (num_points) x (resolution)
            v_list = []
            vectors = vectors.permute(1,0,2)
            for v in vectors:
                    unique_vectors, counts = torch.unique(v, return_counts=True, dim=0)
                    most_common_vector = unique_vectors[counts.argmax()]
                    v_list.append(most_common_vector)

            return torch.stack(v_list)
        
        mode_assignments = _row_mode(self.assignments)
        mode_assignments = mode_assignments.unsqueeze(0)

        # check
        rows_all_zeros = mode_assignments.sum(dim=1) == 0
        indices_all_zeros = torch.nonzero(rows_all_zeros, as_tuple=True)
        print("Indices of rows that are all zeros:", indices_all_zeros)

        # compute mappers
        mode_clusters = compute_mapper(self.data, self.clustering, mode_assignments, input_type=self.type)
        mode_clusters = torch.Tensor(mode_clusters) #{0,1}
        mode_clusters = mode_clusters.to(self.f.dtype)

        # compute filterations
        mode_filtration = compute_filtration(mode_clusters,self.f)
        G_mode = draw_graph(mode_clusters[0],mode_filtration[0], self.path, self.name+"_mode", self.format,save_fig)

        self.mode_assignments = mode_assignments
        self.mode_clusters = mode_clusters
        self.mode_filtration = mode_filtration

        return G_mode
    
    def _row_mode_mutil(self,scheme): # (num_points) x (resolution)
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
    
    def mode(self,save_fig = False):
        
        print("mode:")

        mode_assignments = self._row_mode_mutil(self.scheme)
        self.assignments = mode_assignments

        # check
        rows_all_zeros = mode_assignments.sum(dim=1) == 0
        indices_all_zeros = torch.nonzero(rows_all_zeros, as_tuple=True)
        print("Indices of rows that are all zeros:", indices_all_zeros)
        
        # compute mapper
        mode_assignments = mode_assignments.unsqueeze(0)
        mode_clusters=compute_mapper(self.data, self.clustering, mode_assignments, input_type=self.type)
        mode_clusters = torch.Tensor(mode_clusters) #{0,1}
        mode_clusters = mode_clusters.to(self.f.dtype)

        # compute filtration
        mode_filtration = compute_filtration(mode_clusters,self.f)
        G_mode = draw_graph(mode_clusters[0],mode_filtration[0], self.path, self.name+"_mode", self.format,save_fig)

        self.mode_assignments = mode_assignments
        self.mode_clusters = mode_clusters
        self.mode_filtration = mode_filtration

        return G_mode