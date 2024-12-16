# Implicit-interval-Mapper
This project implements a Mapper Algorithm with implicit intervals and its optimization. All examples in the mannuscript are provided in a Jupyter notebook. The data folder contains the datasets used in the simulation studies, while the model folder contains the implementation of our proposed model.

## Define a Mapper through `Opt_GMM_Mapper` class
You can easily define a GMM soft Mapper using the `Opt_GMM_Mapper` class. The input parameters required are the number of intervals and the initial parameters for a GMM model. We recommend fitting a GMM directly to the projected data and using these fitted parameters as the initial values.

```python
from model import Opt_GMM_Mapper

# Inputs: number of intervals, initial parameters.
# We recommand use direclt fit a GMM on projected data as initial parameters.
m = Opt_GMM_Mapper(n_comp, means = init_mean, covariances=init_var, weights=init_weights)
```

## Compute Mapper graph mode
Computing the Mapper graph mode is straightforward. A single step in the process can yield the distribution Q. Subsequently, the Mapper graph mode can be derived using the `get_mode_graph()` function, the return is a networkx graph.

```python

# Compute the event probability matrix based on the  data, filtered (projected) data, and clustering scheme.
Q = m(projected_data, data, clustering)

# Get the Mapper graph mode
Mapper_graph_mode = m.get_mode_graph()

```
## Optimization
Optimization is a standard process in PyTorch. To simplify the training process, we provide a Trainer class. The `Trainer.fit(data, projected_data, l1, l2)` method automatically completes the training, `l1` and `l2` is weight of loss function. While `Trainer.analysis()` offers a simple analysis of the training process, including the variation of the loss function.

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
We also proved a function to easily sample from the distribution, `Opt_GMM_Mapper.sample(num_samples, Q)`. This function will return a list of networkx graphs.
```python
# Sample 8 samples from optimized distribution
G_list = m.sample(8, train.scheme)

```





