import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 13))
ax.set_xlim(0, 16)
ax.set_ylim(0, 13)
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


draw_box(0.5, 11.5, 2, 0.9, 'INPUT\nx only',
         color='#FFF3E0', border='#FF9800', fontsize=12)

draw_box(4.5, 11.3, 3.5, 1.3, 'Neural Network\n4 layers × 200 neurons\ntanh',
         color='#E8F0FE', border='#4285F4', fontsize=11)

arrow(2.5, 11.95, 4.5, 11.95)

draw_box(10, 12.0, 2.2, 0.6, 'û₁(x)', color='#E8F8E8', border='#34A853', fontsize=11)
draw_box(10, 11.2, 2.2, 0.6, 'û₂(x)', color='#E8F8E8', border='#34A853', fontsize=11)
draw_box(10, 10.4, 2.2, 0.6, 'û₃(x)', color='#C8E6C9', border='#2E7D32', fontsize=11)

ax.text(12.5, 12.3, '← u at t ≈ 0.3 (stage 1)', fontsize=10, va='center', color='#555')
ax.text(12.5, 11.5, '← u at t ≈ 0.7 (stage 2)', fontsize=10, va='center', color='#555')
ax.text(12.5, 10.7, '← u at t = 0.9 (THE ANSWER)', fontsize=10, va='center',
        color='#2E7D32', fontweight='bold')

arrow(8, 12.15, 10, 12.3)
arrow(8, 11.9, 10, 11.5)
arrow(8, 11.6, 10, 10.7)

ax.text(8, 9.7, '▼  autograd: compute û₁_xx and û₂_xx', fontsize=11,
        ha='center', fontweight='bold', color='#C62828')

draw_box(2.5, 8.6, 4.5, 0.8,
         'N₁ = −0.0001·û₁_xx + 5·û₁³ − 5·û₁',
         color='#F3E5F5', border='#7B1FA2', fontsize=10)
draw_box(9, 8.6, 4.5, 0.8,
         'N₂ = −0.0001·û₂_xx + 5·û₂³ − 5·û₂',
         color='#F3E5F5', border='#7B1FA2', fontsize=10)

ax.text(8, 8.0, '▼  RK reconstruction (Butcher tableau weights)',
        fontsize=11, ha='center', fontweight='bold', color='#1565C0')

draw_box(0.5, 6.6, 4.5, 0.9,
         'r₁ = û₁ + Δt·(a₁₁·N₁ + a₁₂·N₂)',
         color='#E3F2FD', border='#1565C0', fontsize=10)
draw_box(5.8, 6.6, 4.5, 0.9,
         'r₂ = û₂ + Δt·(a₂₁·N₁ + a₂₂·N₂)',
         color='#E3F2FD', border='#1565C0', fontsize=10)
draw_box(11, 6.6, 4.2, 0.9,
         'r₃ = û₃ + Δt·(b₁·N₁ + b₂·N₂)',
         color='#E3F2FD', border='#1565C0', fontsize=10)

arrow(4.75, 8.6, 2.75, 7.5, color='#7B1FA2')
arrow(11.25, 8.6, 8.05, 7.5, color='#7B1FA2')
arrow(4.75, 8.6, 8.05, 7.5, color='#7B1FA2')
arrow(11.25, 8.6, 13.1, 7.5, color='#7B1FA2')

ax.text(8, 5.9, '▼  each r should equal the KNOWN data at t = 0.1',
        fontsize=11, ha='center', fontweight='bold', color='#E65100')

draw_box(3, 4.6, 10, 0.9,
         'KNOWN DATA:  u(0.1, xₖ)  at 200 spatial points',
         color='#FFF8E1', border='#F57F17', fontsize=12)

arrow(2.75, 6.6, 5, 5.5, color='#E65100')
arrow(8.05, 6.6, 8, 5.5, color='#E65100')
arrow(13.1, 6.6, 11, 5.5, color='#E65100')

ax.text(8, 3.8, '▼  loss = how far off are the reconstructions?',
        fontsize=11, ha='center', fontweight='bold', color='#B71C1C')

draw_box(1.5, 2.4, 13, 1.1,
         'loss = Σ over 200 points:  (r₁ − uₖ)²  +  (r₂ − uₖ)²  +  (r₃ − uₖ)²',
         color='#FFEBEE', border='#B71C1C', fontsize=13)

ax.text(8, 1.5, 'Train the network to minimize this loss.',
        fontsize=12, ha='center', color='#333', style='italic')
ax.text(8, 0.9, 'Once trained, û₃(x) IS the predicted solution at t = 0.9.',
        fontsize=12, ha='center', color='#2E7D32', fontweight='bold')
ax.text(8, 0.3, 'û₁ and û₂ were just scaffolding — throw them away.',
        fontsize=11, ha='center', color='#888', style='italic')

plt.title('R152: Allen-Cahn Discrete PINN — Simplified (q = 2, only 3 outputs)',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('demos/r152_allen_cahn_simple.png', dpi=150, bbox_inches='tight')
plt.show()
