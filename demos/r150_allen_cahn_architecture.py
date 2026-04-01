import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')


def draw_box(x, y, w, h, text, color='#E8F0FE', border='#4285F4', fontsize=10):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border, linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)


def draw_arrow(x1, y1, x2, y2, label='', color='#333333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, fontsize=8, color='#666666')


draw_box(0.3, 10.2, 2, 1, 'INPUT\nx only\n(no time!)',
         color='#FFF3E0', border='#FF9800', fontsize=11)

draw_box(4, 10.2, 3.5, 1, 'Neural Network\n4 layers × 200 neurons\ntanh activation',
         color='#E8F0FE', border='#4285F4', fontsize=10)

draw_arrow(2.3, 10.7, 4, 10.7)

output_y = 10.0
output_gap = 0.55
outputs = [
    ('û₁(x)', 'u at t ≈ 0.27'),
    ('û₂(x)', 'u at t ≈ 0.50'),
    ('û₃(x)', 'u at t ≈ 0.73'),
    ('û₄(x)', 'u at t = 0.90'),
]

draw_arrow(7.5, 10.7, 9, 10.7)

output_top = 11.0
for idx, (name, desc) in enumerate(outputs):
    oy = output_top - idx * output_gap
    draw_box(9, oy, 1.3, 0.45, name, color='#E8F8E8', border='#34A853', fontsize=9)
    ax.text(10.4, oy + 0.22, f'  ← {desc}', fontsize=8, va='center', color='#555555')

ax.text(14.5, 11.2, 'Toy: q = 3 stages',
        fontsize=10, ha='center', color='#888888', style='italic')
ax.text(14.5, 10.75, 'Real paper: q = 100 stages',
        fontsize=10, ha='center', color='#888888', style='italic')
ax.text(14.5, 10.3, '(101 outputs instead of 4)',
        fontsize=10, ha='center', color='#888888', style='italic')

ax.plot([0.3, 17.5], [9.5, 9.5], color='#C62828', linewidth=1, linestyle='--')
ax.text(9, 9.6, '── autograd: compute u_xx for each stage output ──',
        fontsize=11, ha='center', fontweight='bold', color='#C62828')

deriv_labels = ['û₁_xx', 'û₂_xx', 'û₃_xx', 'û₄_xx']
for idx, label in enumerate(deriv_labels):
    bx = 1.5 + idx * 3.8
    draw_box(bx, 8.5, 1.8, 0.6, label, color='#FCE4EC', border='#C62828', fontsize=10)

ax.plot([0.3, 17.5], [7.8, 7.8], color='#7B1FA2', linewidth=1, linestyle='--')
ax.text(9, 7.9, '── Allen-Cahn operator: N[û_j] = −0.0001·û_j_xx + 5·û_j³ − 5·û_j ──',
        fontsize=10, ha='center', fontweight='bold', color='#7B1FA2')

operator_labels = ['N[û₁]', 'N[û₂]', 'N[û₃]', 'N[û₄]']
for idx, label in enumerate(operator_labels):
    bx = 1.5 + idx * 3.8
    draw_box(bx, 6.9, 1.8, 0.6, label, color='#F3E5F5', border='#7B1FA2', fontsize=10)
    draw_arrow(bx + 0.9, 8.5, bx + 0.9, 7.5, color='#7B1FA2')

ax.plot([0.3, 17.5], [6.2, 6.2], color='#1565C0', linewidth=1, linestyle='--')
ax.text(9, 6.3, '── RK reconstruction: plug into Butcher tableau ──',
        fontsize=11, ha='center', fontweight='bold', color='#1565C0')

rk_labels = [
    'u_n¹ = û₁ + Δt·Σ a₁ⱼ·N[ûⱼ]',
    'u_n² = û₂ + Δt·Σ a₂ⱼ·N[ûⱼ]',
    'u_n³ = û₃ + Δt·Σ a₃ⱼ·N[ûⱼ]',
    'u_n⁴ = û₄ + Δt·Σ bⱼ·N[ûⱼ]',
]
for idx, label in enumerate(rk_labels):
    bx = 0.3 + idx * 4.3
    draw_box(bx, 5.0, 3.9, 0.8, label, color='#E3F2FD', border='#1565C0', fontsize=9)

ax.plot([0.3, 17.5], [4.3, 4.3], color='#E65100', linewidth=1, linestyle='--')
ax.text(9, 4.4, '── each reconstruction should equal the KNOWN data at t = 0.1 ──',
        fontsize=11, ha='center', fontweight='bold', color='#E65100')

draw_box(3, 2.8, 12, 1.1,
         'KNOWN DATA: u(0.1, x) at 200 spatial points\n'
         'Each u_nⁱ is compared against this same data',
         color='#FFF8E1', border='#F57F17', fontsize=11)

for idx in range(4):
    bx = 0.3 + idx * 4.3 + 1.95
    draw_arrow(bx, 5.0, bx, 3.9, color='#E65100')

draw_box(2, 0.5, 6.5, 1.2,
         'loss_data = Σ over all 4 reconstructions,\n'
         'all 200 points: |u_nⁱ(xₖ) − u_known(xₖ)|²',
         color='#FFEBEE', border='#B71C1C', fontsize=10)

draw_box(10, 0.5, 6.5, 1.2,
         'loss_boundary = periodic BCs\n'
         '|û_j(−1) − û_j(1)|² + |û_j_x(−1) − û_j_x(1)|²',
         color='#FFEBEE', border='#B71C1C', fontsize=10)

draw_arrow(9, 2.8, 5.25, 1.7, color='#B71C1C')
draw_arrow(9, 2.8, 13.25, 1.7, color='#B71C1C')

ax.text(9, 0.1, 'total_loss = loss_data + loss_boundary',
        fontsize=13, ha='center', fontweight='bold', color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

plt.title('R150: PINN Discrete Time Architecture for Allen-Cahn\n'
          'Input: x only  |  Output: solution at q intermediate RK stages + final time\n'
          'Physics encoded via RK constraints, not collocation points',
          fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('demos/r150_allen_cahn_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
