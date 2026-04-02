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

center_index = NUM_SPATIAL_POINTS // 2

initial_shape = 2.0 / np.cosh(x)

wave_frequencies = np.fft.fftfreq(NUM_SPATIAL_POINTS, d=dx) * 2 * np.pi
dispersion_half_step = np.exp(-0.5j * 0.5 * wave_frequencies**2 * TIME_STEP)

h = initial_shape.astype(complex)
frames_h_full = [h.copy()]
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
        frames_h_full.append(h.copy())
        frames_amplitude.append(np.abs(h.copy()))
        frames_time.append(step * TIME_STEP)

fig, (ax_wave, ax_complex) = plt.subplots(2, 1, figsize=(10, 8))

line_wave, = ax_wave.plot(x, frames_amplitude[0], 'b-', linewidth=2)
ax_wave.set_xlim(SPATIAL_LEFT, SPATIAL_RIGHT)
ax_wave.set_ylim(0, 4.5)
ax_wave.set_xlabel('Position x')
ax_wave.set_ylabel('|h(t, x)|')
time_label = ax_wave.set_title(f't = {frames_time[0]:.3f}')

h_at_center = frames_h_full[0][center_index]

quiver = ax_complex.quiver(
    0, 0,
    h_at_center.real, h_at_center.imag,
    color='blue', scale=1, angles='xy', scale_units='xy'
)
dot, = ax_complex.plot(h_at_center.real, h_at_center.imag, 'ro', markersize=6)
ax_complex.set_xlim(-3, 3)
ax_complex.set_ylim(-3, 3)
ax_complex.set_aspect('equal')
ax_complex.set_xlabel('Real part (u)')
ax_complex.set_ylabel('Imaginary part (v)')
ax_complex.set_title('h(t, x=0) in the complex plane')
ax_complex.axhline(y=0, color='gray', linewidth=0.3)
ax_complex.axvline(x=0, color='gray', linewidth=0.3)


def update(frame_index):
    line_wave.set_ydata(frames_amplitude[frame_index])
    time_label.set_text(f't = {frames_time[frame_index]:.3f}')

    h_at_center = frames_h_full[frame_index][center_index]
    quiver.set_UVC(h_at_center.real, h_at_center.imag)
    dot.set_data([h_at_center.real], [h_at_center.imag])

    return line_wave, time_label, quiver, dot


animation = FuncAnimation(
    fig,
    update,
    frames=len(frames_amplitude),
    interval=40,
    blit=True
)

plt.tight_layout()
plt.show()
