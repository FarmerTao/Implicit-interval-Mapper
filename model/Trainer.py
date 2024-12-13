from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from .Soft_Mapper import GMM_Soft_Mapper

class Trainer():
    def __init__(self, Mapper,clustering,
                num_step, optimizer, scheduler): #Mapper: Opt_GMM_Mapper
        self.clustering = clustering
        self.Mapper = Mapper
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_step = num_step

        self.losses = None
        self.topo_losses = None
        self.scheme = None
        self.G_mode = None
        self.mode_clusters = None
        self.mode_assignmnets = None
        
    def fit(self, data, projected_data, l1 = 1, l2 = 1): # likelihood, topo
        losses = []
        topo_losses = []
        # train
        for epoch in tqdm(range(self.num_step)):
            
            #overall loss the weights of ll should larger than other terms
            scheme = self.Mapper.forward(projected_data, data, self.clustering)
            
            loss = self.Mapper.loss(l1,l2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            for name, param in self.Mapper.named_parameters():
                if torch.isnan(param.grad).any():
                    print("Warning: The {} gradient was detected as NaN".format(name))

            losses.append(loss.detach().numpy())
            topo_losses.append(self.Mapper.topo_mode_loss.detach().numpy())

        self.losses = losses
        self.topo_losses = topo_losses

        print("loss:",losses[-1])
        print("topo_loss:",topo_losses[-1])

        self.scheme = scheme

        # compute mode
        mapper = GMM_Soft_Mapper(scheme, self.clustering, data = data, projected_data =projected_data,
                                  path='figures',name = 'MSBB_44',format = 'eps',type = self.Mapper.type)
        self.G_mode = mapper.mode(save_fig= False)
        self.mode_clusters = mapper.mode_clusters
        self.mode_assignments = mapper.assignments
        return

    def analysis(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        plt.plot(self.topo_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Topo Loss')
        plt.show()

        self.Mapper.draw()
        return

        

