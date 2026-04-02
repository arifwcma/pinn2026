import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_SPATIAL_POINTS = 512
NUM_TIME_STEPS = 10000
SPATIAL_LEFT = -5.0
SPATIAL_RIGHT = 5.0
TIME_END = np.pi / 2
TIME_STEP = TIME_END / NUM_TIME_STEPS
NUM_FRAMES = 200
SAVE_EVERY = NUM_TIME_STEPS // NUM_FRAMES

x = np.linspace(SPATIAL_LEFT, SPATIAL_RIGHT, NUM_SPATIAL_POINTS, endpoint=False)
dx = x[1] - x[0]

initial_shape = 2.0 / np.cosh(x)

wave_frequencies = np.fft.fftfreq(NUM_SPATIAL_POINTS, d=dx) * 2 * np.pi
dispersion_half_step = np.exp(-0.5j * 0.5 * wave_frequencies**2 * TIME_STEP)

h = initial_shape.astype(complex)
frames_amplitude = [np.abs(h.copy())]
frames_time = [0.0]

for step in range(1, NUM_TIME_STEPS + 1):
    h_fourier = np.fft.fft(h)
    h_fourier = h_fourier * dispersion_half_step
    h = np.fft.ifft(h_fourier)

    h = h * np.exp(1j * np.abs(h)**2 * TIME_STEP)

    h_fourier = np.fft.fft(h)
    h_fourier = h_fourier * dispersion_half_step
    h = np.fft.ifft(h_fourier)

    if step % SAVE_EVERY == 0:
        frames_amplitude.append(np.abs(h.copy()))
        frames_time.append(step * TIME_STEP)

fig, ax = plt.subplots(figsize=(9, 4))
line, = ax.plot(x, frames_amplitude[0], 'b-', linewidth=2)
ax.set_xlim(SPATIAL_LEFT, SPATIAL_RIGHT)
ax.set_ylim(0, 4.5)
ax.set_xlabel('Position x')
ax.set_ylabel('|h(t, x)|')
time_label = ax.set_title(f't = {frames_time[0]:.3f}')


def update(frame_index):
    line.set_ydata(frames_amplitude[frame_index])
    time_label.set_text(f't = {frames_time[frame_index]:.3f}')
    return line, time_label


animation = FuncAnimation(
    fig,
    update,
    frames=len(frames_amplitude),
    interval=40,
    blit=True
)

plt.tight_layout()
plt.show()
