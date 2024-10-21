import graph_tool as gt


from model import Opt_GMM_Mapper
from model import Trainer
from model import Soft_Mapper

import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from gudhi.cover_complex import MapperComplex
import matplotlib.pyplot as plt
import networkx as nx


def off2numpy(shape_name):
    with open(shape_name, 'r') as S:
        S.readline()
        num_vertices, num_faces, _ = [int(n) for n in S.readline().split(' ')]
        info = S.readlines()
    vertices = np.array([[float(coord) for coord in l.split(' ')] for l in info[0:num_vertices]])
    faces    = np.array([[int(coord) for coord in l.split(' ')[1:]] for l in info[num_vertices:]])
    return vertices, faces
    

# data
vertices, faces = off2numpy('data/human.off')

data = torch.tensor(vertices)
x = data[:,0]
y = data[:,1]
z = data[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.grid(None)
ax.axis('off')
plt.show()

