import torch
import torch.nn as nn

# Define model
class Policy(torch.nn.Module):
    def __init__(self, num_actions, num_observations):
        super().__init__()
        # define actor and critic networks

        n_actions = num_actions        
        n_features = num_observations
        layer1_count = 512
        layer2_count = 512
        layer3_count = 512

        
        self.shared1 = nn.Sequential(
                                    nn.Linear(n_features, layer1_count),
                                    nn.ELU()
                                    )
        
        self.shared2 = nn.Sequential(
                                    nn.Linear(layer1_count+n_features, layer2_count),
                                    nn.ELU()
                                    )
        
        self.shared3 = nn.Sequential(
                                    nn.Linear(layer2_count+n_features, layer3_count),
                                    nn.ELU()
                                    )
        
        self.policy1 = nn.Sequential(
                                    nn.Linear(layer3_count, n_actions),
                                    nn.Tanh()
                                    )
        # self.policy2 = nn.Sequential(
        #                             nn.Linear(layer3_count+n_features, n_actions),
        #                             nn.Tanh(),
        #                             )
        
        self.value1 = nn.Sequential(
                                    nn.Linear(layer3_count, 1),
                                    )
        # self.value2 = nn.Sequential(
        #                             nn.Linear(layer2_count+n_features, 1),
        #                             )
        

    def forward(self, x):
        s1 = torch.cat((self.shared1(x), x), dim=-1)
        s2 = torch.cat((self.shared2(s1), x), dim=-1)
        s3 = self.shared3(s2)
        v = self.value1(s3)
        p = self.policy1(s3)
        # v1 = torch.cat((self.value1(s2) , x), dim=-1)
        # v2 = self.value2(v1) 
        # p1 = torch.cat((self.policy1(s2), x), dim=-1)
        # p2 = self.policy2(p1)        

        return v, p