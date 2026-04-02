import numpy as np
import matplotlib.pyplot as plt

NUM_SPATIAL_POINTS = 512
NUM_TIME_STEPS = 10000
SPATIAL_LEFT = -5.0
SPATIAL_RIGHT = 5.0
TIME_END = np.pi / 2
TIME_STEP = TIME_END / NUM_TIME_STEPS

x = np.linspace(SPATIAL_LEFT, SPATIAL_RIGHT, NUM_SPATIAL_POINTS, endpoint=False)
dx = x[1] - x[0]

initial_shape = 2.0 / np.cosh(x)

wave_frequencies = np.fft.fftfreq(NUM_SPATIAL_POINTS, d=dx) * 2 * np.pi
dispersion_half_step = np.exp(-0.5j * 0.5 * wave_frequencies**2 * TIME_STEP)

h = initial_shape.astype(complex)
snapshots = [np.abs(h.copy())]
snapshot_times = [0.0]
num_snapshots = 5
save_every = NUM_TIME_STEPS // num_snapshots

for step in range(1, NUM_TIME_STEPS + 1):
    h_fourier = np.fft.fft(h)
    h_fourier = h_fourier * dispersion_half_step
    h = np.fft.ifft(h_fourier)

    h = h * np.exp(1j * np.abs(h)**2 * TIME_STEP)

    h_fourier = np.fft.fft(h)
    h_fourier = h_fourier * dispersion_half_step
    h = np.fft.ifft(h_fourier)

    if step % save_every == 0:
        snapshots.append(np.abs(h.copy()))
        snapshot_times.append(step * TIME_STEP)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

time_grid, space_grid = np.meshgrid(
    np.linspace(0, TIME_END, num_snapshots + 1),
    x
)
amplitude_grid = np.array(snapshots).T
axes[0].pcolormesh(time_grid, space_grid, amplitude_grid, cmap='plasma', shading='auto')
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('Position x')
axes[0].set_title('|h(t, x)| — Wave amplitude over space and time')

colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
for i, (snapshot, time) in enumerate(zip(snapshots, snapshot_times)):
    axes[1].plot(x, snapshot, color=colors[i], label=f't = {time:.2f}')
axes[1].set_xlabel('Position x')
axes[1].set_ylabel('|h(t, x)|')
axes[1].set_title('Wave shape at different time snapshots')
axes[1].legend()

plt.suptitle('S1: Nonlinear Schrödinger Equation — How the wave evolves', fontsize=13)
plt.tight_layout()
plt.savefig('demos/s1_schrodinger_visualize.png', dpi=150)
plt.show()
