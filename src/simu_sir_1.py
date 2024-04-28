import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

torch.manual_seed(123)
num_samples=500
#simulate beta, gamma
beta_mean, gamma_mean = 0.3, 0.1
beta_var, gamma_var = 1, 1
beta_samples = torch.abs(torch.randn(num_samples) * torch.sqrt(torch.tensor(beta_var)) + beta_mean)
gamma_samples = torch.abs(torch.randn(num_samples) * torch.sqrt(torch.tensor(gamma_var)) + gamma_mean)

# simulate S0, I0, R0
dirichlet_samples = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0])).sample((num_samples,))
class SIRModel(nn.Module):
    def __init__(self, beta, gamma):
        super(SIRModel, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, t, y):
        S, I, R = y[..., 0], y[..., 1], y[..., 2]
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return torch.stack([dSdt, dIdt, dRdt], dim=-1)

t = torch.linspace(0, 100, 200)
results = []

for i in range(num_samples):
    S0, I0, R0 = dirichlet_samples[i]
    beta, gamma = beta_samples[i], gamma_samples[i]
    model = SIRModel(beta, gamma)
    initial_state = torch.tensor([S0, I0, R0], dtype=torch.float32)
    result = odeint(model, initial_state, t)
    results.append(result)

# Convert list of tensors to a single tensor
results_tensor = torch.stack(results)

## Building Neural Network
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return torch.stack([dSdt, dIdt, dRdt])

# Encoder define
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()  # ensure the output is between 0 and 1.
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# Decoder定义
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.initial_time = torch.tensor([0.], dtype=torch.float32)
        self.time_points = torch.linspace(0, 100, 200)  

    def forward(self, params):
        beta, gamma, S0, I0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        R0 = 1 - S0 - I0
        initial_conditions = torch.stack([S0, I0, R0], dim=1)

        solutions = []
        for i in range(initial_conditions.shape[0]):
            solution = odeint(lambda t, y: sir_model(t, y, beta[i], gamma[i]), initial_conditions[i], self.time_points)
            solutions.append(solution)

        return torch.stack(solutions)


# initilize
device = torch.device("cpu")
encoder = Encoder(200 * 3).to(device)
decoder = Decoder().to(device)

# data loading
dataset = TensorDataset(results_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# define loss and optimizer
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
criterion = nn.MSELoss() #L2loss

# training
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0].to(device)  
        params = encoder(inputs)  # get params from encoder
        print(params) # params for each epoch 
        reconstructed = decoder(params)  # get a new solutions

        # calculate loss
        loss = criterion(reconstructed, inputs)  #the loss between input and output solution

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}") #record
