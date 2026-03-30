import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

g = 9.8
h0 = 2.0
v0 = 15.0

def true_h(t):
    return h0 + v0 * t - 0.5 * g * t ** 2

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

t_data = torch.tensor([[0.0], [3.0]])
h_data = true_h(t_data)

t_physics = torch.linspace(0, 3, 200).reshape(-1, 1)
t_physics.requires_grad = True

for epoch in range(5000):
    optimizer.zero_grad()

    loss_data = torch.mean((model(t_data) - h_data) ** 2)

    h_pred = model(t_physics)
    dh = torch.autograd.grad(h_pred, t_physics, torch.ones_like(h_pred), create_graph=True)[0]
    d2h = torch.autograd.grad(dh, t_physics, torch.ones_like(dh), create_graph=True)[0]
    residual = d2h + g
    loss_physics = torch.mean(residual ** 2)

    loss = loss_data + loss_physics
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}: loss_data={loss_data.item():.6f}, loss_physics={loss_physics.item():.6f}")

t_test = torch.linspace(0, 3, 300).reshape(-1, 1)
with torch.no_grad():
    h_pred = model(t_test).numpy()

plt.figure(figsize=(8, 4))
plt.plot(t_test.numpy(), true_h(t_test).numpy(), 'k-', label='True trajectory')
plt.plot(t_test.numpy(), h_pred, 'r--', label='PINN prediction')
plt.scatter([0, 3], [true_h(torch.tensor(0.0)), true_h(torch.tensor(3.0))], c='blue', s=100, zorder=5, label='Training data (2 points)')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.title('R1: Forward Problem — PINN recovers full trajectory from 2 data points + physics')
plt.tight_layout()
plt.savefig('demos/r1_forward_ball.png', dpi=150)
plt.show()
