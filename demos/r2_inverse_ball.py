import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

TRUE_GRAVITY_ON_MARS = 3.72
INITIAL_HEIGHT = 2.0
INITIAL_VELOCITY = 15.0
NUM_OBSERVED_POINTS = 20
NUM_EPOCHS = 10000


def true_height(time):
    return INITIAL_HEIGHT + INITIAL_VELOCITY * time - 0.5 * TRUE_GRAVITY_ON_MARS * time ** 2


class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.discovered_gravity = nn.Parameter(torch.tensor([1.0]))

    def forward(self, time):
        return self.layers(time)


model = PhysicsInformedNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

time_observed = torch.linspace(0, 7, NUM_OBSERVED_POINTS).reshape(-1, 1)
height_observed_exact = true_height(time_observed)
noise = 0.3 * torch.randn_like(height_observed_exact)
height_observed = height_observed_exact + noise

time_collocation = torch.linspace(0, 7, 300).reshape(-1, 1)
time_collocation.requires_grad = True

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    height_predicted_at_data = model(time_observed)
    loss_data = torch.mean((height_predicted_at_data - height_observed) ** 2)

    height_at_collocation = model(time_collocation)

    ones_like_height = torch.ones_like(height_at_collocation)
    velocity = torch.autograd.grad(
        outputs=height_at_collocation,
        inputs=time_collocation,
        grad_outputs=ones_like_height,
        create_graph=True
    )[0]

    ones_like_velocity = torch.ones_like(velocity)
    acceleration = torch.autograd.grad(
        outputs=velocity,
        inputs=time_collocation,
        grad_outputs=ones_like_velocity,
        create_graph=True
    )[0]

    physics_residual = acceleration + model.discovered_gravity
    loss_physics = torch.mean(physics_residual ** 2)

    total_loss = loss_data + loss_physics
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        current_g = model.discovered_gravity.item()
        print(f"Epoch {epoch+1}: loss_data={loss_data.item():.4f}, "
              f"loss_physics={loss_physics.item():.4f}, "
              f"discovered_g={current_g:.4f} (true={TRUE_GRAVITY_ON_MARS})")

print(f"\nFinal discovered gravity: {model.discovered_gravity.item():.4f}")
print(f"True gravity on Mars:     {TRUE_GRAVITY_ON_MARS}")

time_test = torch.linspace(0, 7, 300).reshape(-1, 1)
with torch.no_grad():
    height_predicted = model(time_test).numpy()

plt.figure(figsize=(8, 4))
plt.plot(time_test.numpy(), true_height(time_test).numpy(), 'k-', label='True trajectory')
plt.plot(time_test.numpy(), height_predicted, 'r--', label='PINN prediction')
plt.scatter(
    time_observed.numpy(), height_observed.numpy(),
    c='blue', s=40, zorder=5, label=f'Noisy observations ({NUM_OBSERVED_POINTS} points)'
)
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.title(f'R2: Inverse Problem — Discovered g={model.discovered_gravity.item():.3f} (true={TRUE_GRAVITY_ON_MARS})')
plt.tight_layout()
plt.savefig('demos/r2_inverse_ball.png', dpi=150)
plt.show()
