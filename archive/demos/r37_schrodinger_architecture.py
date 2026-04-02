import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
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
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=8, color='#666666')

draw_box(0.5, 8.2, 2, 1, 'INPUTS\nt, x', color='#FFF3E0', border='#FF9800')

draw_box(4, 8.2, 3.5, 1, 'Neural Network\n5 layers, 100 neurons\ntanh activation',
         color='#E8F0FE', border='#4285F4')

draw_box(9, 8.7, 1.5, 0.5, 'u(t,x)', color='#E8F8E8', border='#34A853')
draw_box(9, 8.0, 1.5, 0.5, 'v(t,x)', color='#E8F8E8', border='#34A853')

draw_arrow(2.5, 8.7, 4, 8.7)
draw_arrow(7.5, 8.9, 9, 8.9)
draw_arrow(7.5, 8.3, 9, 8.3)

draw_box(12, 8.2, 2.5, 1, 'OUTPUTS\nu (real part)\nv (imag part)',
         color='#FFF3E0', border='#FF9800')

draw_arrow(10.5, 8.9, 12, 8.9)
draw_arrow(10.5, 8.3, 12, 8.3)

ax.text(8, 7.3, '── autograd derivatives ──', fontsize=11,
        ha='center', fontweight='bold', color='#C62828')

draw_box(0.5, 5.5, 2.2, 0.8, 'u_t, v_t', color='#FCE4EC', border='#C62828')
draw_box(3.5, 5.5, 2.2, 0.8, 'u_x, v_x', color='#FCE4EC', border='#C62828')
draw_box(6.5, 5.5, 2.5, 0.8, 'u_xx, v_xx', color='#FCE4EC', border='#C62828')
draw_box(10, 5.5, 2.5, 0.8, 'u² + v²', color='#FCE4EC', border='#C62828')

draw_arrow(9.75, 8.2, 1.6, 6.3, color='#C62828')
draw_arrow(9.75, 8.2, 4.6, 6.3, color='#C62828')
draw_arrow(9.75, 8.2, 7.75, 6.3, color='#C62828')
draw_arrow(9.75, 8.2, 11.25, 6.3, color='#C62828')

ax.text(8, 4.7, '── assemble two real residuals ──', fontsize=11,
        ha='center', fontweight='bold', color='#7B1FA2')

draw_box(1, 3.2, 5.5, 0.9, 'f_real = -v_t + 0.5·u_xx + (u²+v²)·u',
         color='#F3E5F5', border='#7B1FA2')
draw_box(8, 3.2, 5.5, 0.9, 'f_imag = u_t + 0.5·v_xx + (u²+v²)·v',
         color='#F3E5F5', border='#7B1FA2')

ax.text(8, 2.3, '── three loss terms ──', fontsize=11,
        ha='center', fontweight='bold', color='#1565C0')

draw_box(0.3, 0.8, 3.5, 1,
         'loss_initial\n|h(0,x) - 2sech(x)|²\n(50 points)',
         color='#E3F2FD', border='#1565C0')

draw_box(5, 0.8, 4.5, 1,
         'loss_boundary\n|h(t,-5) - h(t,5)|²\n+ |h_x(t,-5) - h_x(t,5)|²\n(50 time points)',
         color='#E3F2FD', border='#1565C0')

draw_box(10.5, 0.8, 4.5, 1,
         'loss_physics\nmean(f_real²) + mean(f_imag²)\n(20,000 collocation points)',
         color='#E3F2FD', border='#1565C0')

draw_arrow(3.75, 3.2, 12.75, 1.8, color='#1565C0')
draw_arrow(10.75, 3.2, 12.75, 1.8, color='#1565C0')

ax.text(8, 0.3, 'total_loss = loss_initial + loss_boundary + loss_physics',
        fontsize=12, ha='center', fontweight='bold', color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

plt.title('R37: PINN Architecture for Schrödinger Equation — End to End',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('demos/r37_schrodinger_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
