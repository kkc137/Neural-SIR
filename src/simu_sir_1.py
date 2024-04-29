import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint


torch.manual_seed(123)

# Configurable parameters
num_samples = 500
beta_mean, gamma_mean = 0.3, 0.1
beta_var, gamma_var = 1, 1
time_span = 100
num_time_points = 200
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Simulate beta, gamma
beta_samples = torch.abs(torch.randn(num_samples) * torch.sqrt(torch.tensor(beta_var)) + beta_mean)
gamma_samples = torch.abs(torch.randn(num_samples) * torch.sqrt(torch.tensor(gamma_var)) + gamma_mean)

# Simulate S0, I0, R0 (more realistic initial conditions)
S0 = torch.rand(num_samples) * 0.9 + 0.1  # Ensure S0 is never too small
I0 = torch.rand(num_samples) * 0.1  # Assume small initial infections
R0 = torch.zeros(num_samples)  # Assume no initial recovered

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

t = torch.linspace(0, time_span, num_time_points)
results = []

for i in range(num_samples):
    beta, gamma = beta_samples[i], gamma_samples[i]
    model = SIRModel(beta, gamma)
    initial_state = torch.tensor([S0[i], I0[i], R0[i]], dtype=torch.float32)
    result = odeint(model, initial_state, t)
    results.append(result)

# Convert list of tensors to a single tensor
results_tensor = torch.stack(results)

# Normalize input data
results_tensor = (results_tensor - results_tensor.mean()) / results_tensor.std()

# Encoder define
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Sigmoid()  # ensure the output is between 0 and 1.
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Decoder定义
class Decoder(nn.Module):
    def __init__(self, time_span, num_time_points):
        super(Decoder, self).__init__()
        self.initial_time = torch.tensor([0.], dtype=torch.float32)
        self.time_points = torch.linspace(0, time_span, num_time_points)

    def forward(self, params):
        beta, gamma, S0, I0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        R0 = 1 - S0 - I0
        initial_conditions = torch.stack([S0, I0, R0], dim=1)

        solutions = []
        for i in range(initial_conditions.shape[0]):
            solution = odeint(lambda t, y: sir_model(t, y, beta[i], gamma[i]), initial_conditions[i], self.time_points)
            solutions.append(solution)

        return torch.stack(solutions)

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return torch.stack([dSdt, dIdt, dRdt])

# Initialize models
device = torch.device("cpu")
encoder = Encoder(num_time_points * 3).to(device)
decoder = Decoder(time_span, num_time_points).to(device)

# Data loading
dataset = TensorDataset(results_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function (consider using a more appropriate loss for SIR model)
criterion = nn.MSELoss()
# Define optimizer
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
# Initialize a list to store the average loss per epoch
epoch_loss_values = []

# Training loop
for epoch in range(num_epochs):
    # Initialize a variable to store the sum of batch losses
    epoch_loss = 0.0
    num_batches = 0

    for data in dataloader:
        inputs = data[0].to(device)
        params = encoder(inputs)  # get params from encoder
        reconstructed = decoder(params)  # get a new solutions

        # Calculate loss
        loss = criterion(reconstructed, inputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add the loss value for this batch to the epoch's loss sum
        epoch_loss += loss.item()
        num_batches += 1

    # Calculate the average loss for this epoch and store it
    avg_loss = epoch_loss / num_batches
    epoch_loss_values.append(avg_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Create a line plot of average loss values per epoch
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()

# Print lengths for debugging
print(len(range(1, num_epochs + 1)))
print(len(epoch_loss_values))

def plot_param_distributions(params):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # Plot the distribution of each parameter
    for i, ax in enumerate(axs.flatten()):
        sns.histplot(params[:, i], ax=ax, kde=True)
        ax.set_title(f"Parameter {i + 1}")

    plt.tight_layout()
    plt.show()

# Call this function after training, passing the output of the encoder
params = encoder(results_tensor.to(device))
plot_param_distributions(params.detach().cpu().numpy())

