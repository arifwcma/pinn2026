import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

GRAVITY = 9.8
INITIAL_HEIGHT = 2.0
INITIAL_VELOCITY = 15.0

def true_height(time):
    return INITIAL_HEIGHT + INITIAL_VELOCITY * time - 0.5 * GRAVITY * time ** 2

class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, time):
        return self.layers(time)

model = PhysicsInformedNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

time_observed = torch.tensor([[0.0], [3.0]])
height_observed = true_height(time_observed)

time_collocation = torch.linspace(0, 3, 200).reshape(-1, 1)
time_collocation.requires_grad = True

for epoch in range(5000):
    optimizer.zero_grad()

    loss_data = torch.mean((model(time_observed) - height_observed) ** 2)

    height_at_collocation = model(time_collocation)
    velocity = torch.autograd.grad(height_at_collocation, time_collocation, torch.ones_like(height_at_collocation), create_graph=True)[0]
    acceleration = torch.autograd.grad(velocity, time_collocation, torch.ones_like(velocity), create_graph=True)[0]
    physics_residual = acceleration + GRAVITY
    loss_physics = torch.mean(physics_residual ** 2)

    total_loss = loss_data + loss_physics
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}: loss_data={loss_data.item():.6f}, loss_physics={loss_physics.item():.6f}")

time_test = torch.linspace(0, 3, 300).reshape(-1, 1)
with torch.no_grad():
    height_predicted = model(time_test).numpy()

plt.figure(figsize=(8, 4))
plt.plot(time_test.numpy(), true_height(time_test).numpy(), 'k-', label='True trajectory')
plt.plot(time_test.numpy(), height_predicted, 'r--', label='PINN prediction')
plt.scatter([0, 3], [true_height(torch.tensor(0.0)), true_height(torch.tensor(3.0))], c='blue', s=100, zorder=5, label='Training data (2 points)')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.title('R1: Forward Problem — PINN recovers full trajectory from 2 data points + physics')
plt.tight_layout()
plt.savefig('demos/r1_forward_ball.png', dpi=150)
plt.show()
