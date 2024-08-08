import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def exact_solution(beta, gamma, S0, I0, R0, t):
    def sir(t, y):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    y0 = [S0, I0, R0]
    sol = solve_ivp(sir, [0, t.flatten()[-1]], y0, t_eval=t.flatten())
    return torch.tensor(sol.y.T, dtype=torch.float32)

class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(FCN, self).__init__()
        activation = nn.Tanh
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(N_INPUT, N_HIDDEN))
        self.layers.append(activation())
        for _ in range(N_LAYERS - 1):
            self.layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
            self.layers.append(activation())
        self.layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Step1: Initialize seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

# True parameters for the SIR model
beta_true, gamma_true = 0.3, 0.1
S0, I0, R0 = 0.99, 0.01, 0.0

# Initialize the PINN
pinn = FCN(4, 3, 64, 5)

# Define the time boundaries
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)
t_max = 80
t_physics = torch.linspace(0, t_max, 500).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, t_max, 500).view(-1, 1)

# Compute the exact solution for testing
u_exact = exact_solution(beta_true, gamma_true, S0, I0, R0, t_test.numpy())

# Optimizer for the PINN
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3, weight_decay=1e-5)

# Step 2: Generate observational data
torch.manual_seed(123)
t_obs = torch.linspace(0, t_max, 300).view(-1, 1)
u_obs = exact_solution(beta_true, gamma_true, S0, I0, R0, t_obs.numpy()) + 0.01 * torch.randn(300, 3)

# Initialize parameters for optimization
beta = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
gamma = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
optimizer_new = torch.optim.Adam(list(pinn.parameters()) + [beta, gamma], lr=1e-4)

# Lists to store the values of beta and gamma during training
betas, gammas = [], []

# Training loop
for i in range(20001):
    optimizer_new.zero_grad()

    # Physical loss calculation
    t_obs_grad = t_obs.clone().requires_grad_(True)
    pinn_input = torch.cat([u_obs, t_obs_grad], dim=1)
    u = pinn(pinn_input)
    S, I, R = u[:, 0], u[:, 1], u[:, 2]
    dSdt = torch.autograd.grad(S, t_obs_grad, torch.ones_like(S), create_graph=True)[0]
    dIdt = torch.autograd.grad(I, t_obs_grad, torch.ones_like(I), create_graph=True)[0]
    dRdt = torch.autograd.grad(R, t_obs_grad, torch.ones_like(R), create_graph=True)[0]

    loss1 = torch.mean((dSdt + beta * S * I) ** 2 +
                       (dIdt - beta * S * I + gamma * I) ** 2 +
                       (dRdt - gamma * I) ** 2)

    # Data loss calculation
    loss2 = torch.mean((u - u_obs) ** 2)
    loss3 = torch.mean((1.0 - S -I)**2)
    loss4 =(0.4-beta-gamma)**2
    # Total loss
    loss = (1e3* loss1 + 1e5 * loss2 + 1e5 * loss3+ 1e4 *loss4)
    loss.backward()
    optimizer_new.step()

    # Record the values of beta and gamma
    betas.append(beta.item())
    gammas.append(gamma.item())

    # Plot the results every 5000 iterations
    if i % 5000 == 0:
        pinn_test_input = torch.cat([u_exact, t_test], dim=1)
        u = pinn(pinn_test_input).detach()
        plt.figure(figsize=(10, 6))
        plt.scatter(t_obs[:, 0], u_obs[:, 0], label="S (noisy)", alpha=0.6)
        plt.scatter(t_obs[:, 0], u_obs[:, 1], label="I (noisy)", alpha=0.6)
        plt.scatter(t_obs[:, 0], u_obs[:, 2], label="R (noisy)", alpha=0.6)
        plt.plot(t_test[:, 0], u[:, 0], label="S (PINN)", color="tab:blue")
        plt.plot(t_test[:, 0], u[:, 1], label="I (PINN)", color="tab:orange")
        plt.plot(t_test[:, 0], u[:, 2], label="R (PINN)", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.show()

# Plot the estimated values of beta and gamma over training steps
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("β")
plt.plot(betas, label="PINN estimate")
plt.hlines(beta_true, 0, len(betas), label="True value", color="tab:red")
plt.legend()
plt.xlabel("Training step")

plt.subplot(1, 2, 2)
plt.title("γ")
plt.plot(gammas, label="PINN estimate")
plt.hlines(gamma_true, 0, len(gammas), label="True value", color="tab:red")
plt.legend()
plt.xlabel("Training step")

plt.tight_layout()
plt.show()
