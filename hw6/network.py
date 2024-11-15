import torch
import torch.nn as nn

class ClassificationNetworkColors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # setting device on GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classes = [
            [-1.0, 0.0, 0.0],  # 0 left
            [-1.0, 0.5, 0.0],  # 1 left and accelerate
            [-1.0, 0.0, 0.8],  # 2 left and brake
            [1.0, 0.0, 0.0],  # 3 right
            [1.0, 0.5, 0.0],  # 4 right and accelerate
            [1.0, 0.0, 0.8],  # 5 right and brake
            [0.0, 0.0, 0.0],  # 6 no input
            [0.0, 0.5, 0.0],  # 7 accelerate
            [0.0, 0.0, 0.8],  # 8 brake
        ]

        """
        D : Network Implementation

        Implementation of the network layers. 
        The image size of the input observations is 96x96 pixels.

        Using torch.nn.Sequential(), implement each convolution layers and Linear layers
        """
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), # (batch_size, 8, 96, 96)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 8, 48, 48)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # (batch_size, 16, 48, 48)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (batch_size, 16, 24, 24)
        ).to(device)
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 24 * 24, 16 * 24 * 4),
            nn.ELU(),
            nn.Linear(16 * 24 * 4, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, len(self.classes))
        ).to(device)

    def forward(self, observation):
        """
        D : Network Implementation

        The forward pass of the network.
        Returns the prediction for the given input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        # rearrange from (batch_size, 96, 96, 3) to (batch_size, 3, 96, 96)
        x = observation.permute(0, 3, 1, 2) 
        
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc_layers(x)
        return x

    def actions_to_classes(self, actions):
        """
        C : Conversion from action to classes

        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        ret = []

        for action in actions:
            # find the index of the corresponding class
            action_list = action.tolist()
            rounded_action_list = [round(a, 2) for a in action_list]
            index = self.classes.index(rounded_action_list)

            # create a tensor with the index as the only element
            ret.append(torch.tensor([index]))

        return ret

    def scores_to_action(self, scores):
        """
        C : Selection of action from scores

        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C <- list?? went for Tensor of size C
        return          (float, float, float)
        """

        max_idx = int(torch.argmax(scores).item())
        return tuple(self.classes[max_idx])
