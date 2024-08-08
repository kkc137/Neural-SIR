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
    sol = solve_ivp(sir, [0, t[-1, 0]], y0, t_eval=t.flatten())
    return torch.tensor(sol.y.T, dtype=torch.float32)


class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
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


# Step1
torch.manual_seed(123)
np.random.seed(123)

# Define SIR model parameters and initial conditions
beta_true, gamma_true = 0.3, 0.1
S0, I0, R0 = 0.99, 0.01, 0.0

# Define the neural network (increase complexity)
pinn = FCN(4, 3, 64, 5)

# Define boundary points (initial conditions)
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

# Define training points over the entire domain (increase number of points)
t_max = 80
t_physics = torch.linspace(0, t_max, 500).view(-1, 1).requires_grad_(True)

# Generate exact solution for comparison
t_test = torch.linspace(0, t_max, 500).view(-1, 1)
u_exact = exact_solution(beta_true, gamma_true, S0, I0, R0, t_test.numpy())


# Define optimizer and scheduler
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5, verbose=True)


# Step 2
torch.manual_seed(123)
t_obs = torch.linspace(0, t_max, 300).view(-1, 1)
u_obs = exact_solution(beta_true, gamma_true, S0, I0, R0, t_obs.numpy()) + 0.01 * torch.randn(300, 3)

plt.figure(figsize=(10, 6))
plt.title("Noisy observational data")
plt.scatter(t_obs[:, 0], u_obs[:, 0], label="S (noisy)", alpha=0.6)
plt.scatter(t_obs[:, 0], u_obs[:, 1], label="I (noisy)", alpha=0.6)
plt.scatter(t_obs[:, 0], u_obs[:, 2], label="R (noisy)", alpha=0.6)
plt.plot(t_test[:, 0], u_exact[:, 0], label="S (exact)", color="tab:blue", alpha=0.6)
plt.plot(t_test[:, 0], u_exact[:, 1], label="I (exact)", color="tab:orange", alpha=0.6)
plt.plot(t_test[:, 0], u_exact[:, 2], label="R (exact)", color="tab:green", alpha=0.6)
plt.legend()
plt.show()

beta= torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
gamma = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
optimizer_new = torch.optim.Adam(list(pinn.parameters()) + [beta, gamma], lr=1e-4)

betas, gammas = [], []

for i in range(5001):
    optimizer_new.zero_grad()
    # physical loss
    u = pinn(u_obs[0],u_obs[1],u_obs[2],t_obs)
    S, I, R = u[:, 0], u[:, 1], u[:, 2]
    dSdt = torch.autograd.grad(S, t_obs, torch.ones_like(S), create_graph=True)[0]
    dIdt = torch.autograd.grad(I, t_obs, torch.ones_like(I), create_graph=True)[0]
    dRdt = torch.autograd.grad(R, t_obs, torch.ones_like(R), create_graph=True)[0]

    loss1 = torch.mean((dSdt + beta * S * I) ** 2 +
                       (dIdt - beta * S * I + gamma * I) ** 2 +
                       (dRdt - gamma * I) ** 2)

    #  data loss
    loss2 = torch.mean((u - u_obs) ** 2)

    # loss3 = (0.4-beta - gamma) ** 2
    # backpropagate joint loss
    loss = (loss1 + 1e4 * loss2)  #+ 1e3 * loss3)
    loss.backward()
    optimizer_new.step()
    # record gamma beta
    betas.append(beta.item())
    gammas.append(gamma.item())

    if i % 5000 == 0:
        u = pinn(t_test).detach()
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