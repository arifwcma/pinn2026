import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

VISCOSITY = 0.01 / np.pi
NUM_INITIAL_POINTS = 50
NUM_BOUNDARY_POINTS = 25
NUM_COLLOCATION_POINTS = 10000
NUM_EPOCHS = 5000


def exact_solution(time, position):
    numerator = -torch.sin(np.pi * position)
    denominator = 1 + (time / (2 * np.pi * VISCOSITY)) * (1 - torch.cos(np.pi * position))
    return numerator / denominator


class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, time, position):
        inputs = torch.cat([time, position], dim=1)
        return self.layers(inputs)


def compute_physics_residual(model, time, position):
    time.requires_grad_(True)
    position.requires_grad_(True)

    u = model(time, position)

    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(
        outputs=u, inputs=time,
        grad_outputs=ones, create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        outputs=u, inputs=position,
        grad_outputs=ones, create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=position,
        grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    residual = u_t + u * u_x - VISCOSITY * u_xx
    return residual


model = PhysicsInformedNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

position_initial = torch.rand(NUM_INITIAL_POINTS, 1) * 2 - 1
time_initial = torch.zeros(NUM_INITIAL_POINTS, 1)
u_initial = -torch.sin(np.pi * position_initial)

time_boundary = torch.rand(NUM_BOUNDARY_POINTS, 1)
position_left = -torch.ones(NUM_BOUNDARY_POINTS, 1)
position_right = torch.ones(NUM_BOUNDARY_POINTS, 1)
u_boundary_left = torch.zeros(NUM_BOUNDARY_POINTS, 1)
u_boundary_right = torch.zeros(NUM_BOUNDARY_POINTS, 1)

time_observed = torch.cat([time_initial, time_boundary, time_boundary])
position_observed = torch.cat([position_initial, position_left, position_right])
u_observed = torch.cat([u_initial, u_boundary_left, u_boundary_right])

time_collocation = torch.rand(NUM_COLLOCATION_POINTS, 1)
position_collocation = torch.rand(NUM_COLLOCATION_POINTS, 1) * 2 - 1

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    u_predicted_at_data = model(time_observed, position_observed)
    loss_data = torch.mean((u_predicted_at_data - u_observed) ** 2)

    physics_residual = compute_physics_residual(
        model, time_collocation, position_collocation
    )
    loss_physics = torch.mean(physics_residual ** 2)

    total_loss = loss_data + loss_physics
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}: loss_data={loss_data.item():.6f}, loss_physics={loss_physics.item():.6f}")

time_grid, position_grid = torch.meshgrid(
    torch.linspace(0, 1, 100),
    torch.linspace(-1, 1, 256),
    indexing='ij'
)
time_flat = time_grid.reshape(-1, 1)
position_flat = position_grid.reshape(-1, 1)

with torch.no_grad():
    u_predicted = model(time_flat, position_flat).reshape(100, 256).numpy()
    u_exact = exact_solution(time_flat, position_flat).reshape(100, 256).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(u_exact.T, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='seismic')
axes[0].set_title("Exact solution")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Position x")
axes[1].imshow(u_predicted.T, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='seismic')
axes[1].set_title("PINN prediction")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Position x")
plt.suptitle("R10: Burgers' Equation — Forward Problem (Continuous Time PINN)")
plt.tight_layout()
plt.savefig('demos/r10_burgers_forward.png', dpi=150)
plt.show()
