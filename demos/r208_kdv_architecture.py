import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(18, 15))
ax.set_xlim(0, 18)
ax.set_ylim(0, 15)
ax.axis('off')


def draw_box(x, y, w, h, text, color='#E8F0FE', border='#4285F4', fontsize=10):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border, linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')


def arrow(x1, y1, x2, y2, color='#333333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))


draw_box(0.5, 13.5, 2, 0.9, 'INPUT\nx only',
         color='#FFF3E0', border='#FF9800', fontsize=12)

draw_box(4.5, 13.3, 3.5, 1.3, 'Neural Network\n4 layers × 50 neurons\ntanh',
         color='#E8F0FE', border='#4285F4', fontsize=11)

arrow(2.5, 13.95, 4.5, 13.95)

draw_box(10, 14.0, 2.2, 0.6, 'û₁(x)', color='#E8F8E8', border='#34A853', fontsize=11)
draw_box(10, 13.2, 2.2, 0.6, 'û₂(x)', color='#C8E6C9', border='#2E7D32', fontsize=11)

ax.text(12.5, 14.3, '← wave height at t ≈ 0.35 (stage 1)', fontsize=9, va='center', color='#555')
ax.text(12.5, 13.5, '← wave height at t ≈ 0.65 (stage 2)', fontsize=9, va='center', color='#555')

arrow(8, 14.15, 10, 14.3)
arrow(8, 13.6, 10, 13.5)

draw_box(14, 13.5, 2.5, 0.9, 'TRAINABLE\nλ₁ and λ₂\n(start random)',
         color='#FFE0B2', border='#E65100', fontsize=9)

ax.text(9, 12.5, '▼  SLOPE LAYER: autograd gives û₁_x, û₁_xxx, û₂_x, û₂_xxx',
        fontsize=10, ha='center', fontweight='bold', color='#C62828')

draw_box(0.5, 11.2, 7, 0.8,
         'slope₁ = −λ₁·û₁·û₁_x − λ₂·û₁_xxx',
         color='#F3E5F5', border='#7B1FA2', fontsize=10)
draw_box(9, 11.2, 7, 0.8,
         'slope₂ = −λ₁·û₂·û₂_x − λ₂·û₂_xxx',
         color='#F3E5F5', border='#7B1FA2', fontsize=10)

ax.text(8.5, 11.65, '(u_t from KdV equation, uses unknown λ₁ and λ₂)',
        fontsize=8, ha='center', color='#7B1FA2', style='italic')

ax.text(9, 10.4, '▼  RK LAYER: compute RK-predicted u_n and RK-predicted u_{n+1}',
        fontsize=10, ha='center', fontweight='bold', color='#1565C0')

draw_box(0.3, 8.8, 8, 1.2,
         'RK-predicted u_n (should match data at t = 0.2)\n\n'
         'predicted_un_1 = û₁ − Δt·(a₁₁·slope₁ + a₁₂·slope₂)\n'
         'predicted_un_2 = û₂ − Δt·(a₂₁·slope₁ + a₂₂·slope₂)',
         color='#E3F2FD', border='#1565C0', fontsize=9)

draw_box(9.5, 8.8, 8, 1.2,
         'RK-predicted u_{n+1} (should match data at t = 0.8)\n\n'
         'predicted_un1_1 = û₁ − Δt·((a₁₁−b₁)·slope₁ + (a₁₂−b₂)·slope₂)\n'
         'predicted_un1_2 = û₂ − Δt·((a₂₁−b₁)·slope₁ + (a₂₂−b₂)·slope₂)',
         color='#E3F2FD', border='#1565C0', fontsize=9)

ax.text(9, 8.15, '▼  compare against KNOWN DATA at BOTH snapshots',
        fontsize=10, ha='center', fontweight='bold', color='#E65100')

draw_box(0.3, 6.8, 7.5, 0.9,
         'KNOWN DATA at t = 0.2\nwave height at 199 points',
         color='#FFF8E1', border='#F57F17', fontsize=11)

draw_box(10, 6.8, 7.5, 0.9,
         'KNOWN DATA at t = 0.8\nwave height at 201 points',
         color='#FFF8E1', border='#F57F17', fontsize=11)

arrow(4.3, 8.8, 4.05, 7.7, color='#E65100')
arrow(13.5, 8.8, 13.75, 7.7, color='#E65100')

ax.text(9, 6.1, '▼  LOSS LAYER',
        fontsize=11, ha='center', fontweight='bold', color='#B71C1C')

draw_box(0.3, 4.6, 7.5, 1.1,
         'loss_snapshot1 =\nΣ (predicted_un − actual_un)²\nover all 199 points × 2 stages',
         color='#FFEBEE', border='#B71C1C', fontsize=10)

draw_box(10, 4.6, 7.5, 1.1,
         'loss_snapshot2 =\nΣ (predicted_un1 − actual_un1)²\nover all 201 points × 2 stages',
         color='#FFEBEE', border='#B71C1C', fontsize=10)

ax.text(9, 3.8, 'total_loss = loss_snapshot1 + loss_snapshot2',
        fontsize=13, ha='center', fontweight='bold', color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

ax.text(9, 2.8, 'Backpropagation updates TWO things simultaneously:',
        fontsize=11, ha='center', fontweight='bold', color='#333')

draw_box(1, 1.8, 7, 0.7,
         '1. Network weights (to improve û guesses)',
         color='#E8F0FE', border='#4285F4', fontsize=10)

draw_box(10, 1.8, 7, 0.7,
         '2. λ₁ and λ₂ (to discover the PDE coefficients)',
         color='#FFE0B2', border='#E65100', fontsize=10)

ax.text(9, 1.1, 'After training: λ₁ ≈ 1.0, λ₂ ≈ 0.0025 (the true values, discovered from data alone!)',
        fontsize=11, ha='center', color='#2E7D32', fontweight='bold')

plt.title('R208: KdV Inverse Discrete PINN (q = 2)\n'
          'Same as Allen-Cahn but: (1) two snapshots instead of one, (2) λ₁ and λ₂ are trainable',
          fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('demos/r208_kdv_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
