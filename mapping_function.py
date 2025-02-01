import torch.nn as nn

class MLP_old(nn.Module):
    def __init__(self, input_size=768, hidden_size=1024, output_size=768):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)


class Linear_MLP(nn.Module):
    def __init__(self, input_size=768, hidden_size=1024, output_size=768):
        super(Linear_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim = 5 * output_dim  # Hidden layer dimension as 5x output dimension

        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # SELU activation
        self.activation = nn.SELU()

        # L2 normalization for the output
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = self.l2_norm(x, p=2, dim=1)  # Apply L2 normalization along the feature dimension
        return x


class Fix_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Fix_MLP, self).__init__()
        hidden_dim = 3840  

        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # SELU activation
        self.activation = nn.SELU()

        # L2 normalization for the output
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = self.l2_norm(x, p=2, dim=1)  # Apply L2 normalization along the feature dimension
        return x

class Big_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Big_MLP, self).__init__()
        hidden_dim = 5120

        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # SELU activation
        self.activation = nn.SELU()

        # L2 normalization for the output
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = self.l2_norm(x, p=2, dim=1)  # Apply L2 normalization along the feature dimension
        return x


class Costom_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Costom_MLP, self).__init__()

        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # SELU activation
        self.activation = nn.SELU()

        # L2 normalization for the output
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = self.l2_norm(x, p=2, dim=1)  # Apply L2 normalization along the feature dimension
        return x