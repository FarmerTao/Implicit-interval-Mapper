# Implicit-interval-Mapper
This project implements a Mapper Algorithm with implicit intervals and its optimization. All examples from the paper are provided in Jupyter notebooks. The data folder contains the datasets used in these examples, while the model folder contains the implementation of our model.

## Define a Mapper through Opt_GMM_Mapper class
You can easily define a GMM soft Mapper using the Opt_GMM_Mapper class. The input parameters required are the number of intervals and the initial parameters for the GMM model. We recommend fitting a GMM directly to the projected data and using these fitted parameters as the initial values.

```python
from model import Opt_GMM_Mapper

# Inputs: number of intervals, initial parameters.
# We recommand use direclt fit a GMM on projected data as initial parameters.
m = Opt_GMM_Mapper(n_comp, means = init_mean, covariances=init_var, weights=init_weights)
```

## Compute Mapper graph mode
```python

# Define your data, projected data and clustering
Q = m(projected_data, data, clustering)

# Get Mapper graph mode
Mapper_graph_mode = m.get_mode_graph()

```
## Optimization

```python

import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from model import Trainer

num_step = 200
l1 = 1 
l2 = 1 

# define optimizer and scheduler
optimizer = optim.SGD(m.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# mapper after optimization
Mapper_graph_mode_after = m.get_mode_graph()

# training
train = Trainer(m, clustering, num_step, optimizer, scheduler)
train.fit(data, projected_data, l1, l2)
train.analysis()

```


## Sample from the distribution
```python
# Sample 8 samples from optimized distribution
G_list = m.sample(8, train.scheme)

```





