import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')


def draw_box(x, y, w, h, text, color='#E8F0FE', border='#4285F4', fontsize=9):
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


draw_box(0.5, 10, 2, 1, 'INPUTS\nt, x, y', color='#FFF3E0', border='#FF9800')

draw_box(4, 10, 3.5, 1, 'Neural Network\n9 layers, 20 neurons\ntanh activation',
         color='#E8F0FE', border='#4285F4')

draw_box(9, 10.3, 1.8, 0.5, '\u03c8(t,x,y)', color='#E8F8E8', border='#34A853')
draw_box(9, 9.6, 1.8, 0.5, 'p(t,x,y)', color='#E8F8E8', border='#34A853')

draw_arrow(2.5, 10.5, 4, 10.5)
draw_arrow(7.5, 10.7, 9, 10.55)
draw_arrow(7.5, 10.3, 9, 9.85)

ax.text(9, 8.8, '-- stream function trick (autograd) --', fontsize=11,
        ha='center', fontweight='bold', color='#C62828')

draw_box(5.5, 7.5, 3, 0.8, 'u = \u03c8_y      v = -\u03c8_x\n(incompressibility guaranteed)',
         color='#FCE4EC', border='#C62828')

draw_arrow(9.9, 10.3, 7, 8.3, color='#C62828')

ax.text(9, 6.8, '-- autograd: all derivatives --', fontsize=11,
        ha='center', fontweight='bold', color='#C62828')

draw_box(0.3, 5.5, 2, 0.7, 'u_t, v_t', color='#FCE4EC', border='#C62828')
draw_box(2.8, 5.5, 2.5, 0.7, 'u_x, u_y\nv_x, v_y', color='#FCE4EC', border='#C62828')
draw_box(5.8, 5.5, 2.5, 0.7, 'u_xx, u_yy\nv_xx, v_yy', color='#FCE4EC', border='#C62828')
draw_box(8.8, 5.5, 2, 0.7, 'p_x, p_y', color='#FCE4EC', border='#C62828')

draw_box(12, 7.5, 2.5, 1.2, 'TRAINABLE\nPARAMETERS\n\u03bb\u2081, \u03bb\u2082\n(nn.Parameter)',
         color='#FFF9C4', border='#F9A825')

ax.text(9, 4.7, '-- assemble two momentum residuals --', fontsize=11,
        ha='center', fontweight='bold', color='#7B1FA2')

draw_box(0.5, 3.2, 7.5, 0.9,
         'f = u_t + \u03bb\u2081(u\u00b7u_x + v\u00b7u_y) + p_x - \u03bb\u2082(u_xx + u_yy)',
         color='#F3E5F5', border='#7B1FA2')
draw_box(9, 3.2, 7.5, 0.9,
         'g = v_t + \u03bb\u2081(u\u00b7v_x + v\u00b7v_y) + p_y - \u03bb\u2082(v_xx + v_yy)',
         color='#F3E5F5', border='#7B1FA2')

ax.text(9, 2.3, '-- two loss terms --', fontsize=11,
        ha='center', fontweight='bold', color='#1565C0')

draw_box(1, 0.8, 6, 1,
         'loss_data\n|u_pred - u_obs|\u00b2 + |v_pred - v_obs|\u00b2\n(5,000 velocity measurements)',
         color='#E3F2FD', border='#1565C0')

draw_box(9, 0.8, 6.5, 1,
         'loss_physics\nmean(f\u00b2) + mean(g\u00b2)\n(evaluated at same 5,000 points)',
         color='#E3F2FD', border='#1565C0')

draw_arrow(4.25, 3.2, 12.25, 1.8, color='#1565C0')
draw_arrow(12.75, 3.2, 12.25, 1.8, color='#1565C0')

ax.text(9, 0.15, 'total_loss = loss_data + loss_physics    \u2192    learns: network weights + \u03bb\u2081 + \u03bb\u2082 + p(t,x,y)',
        fontsize=11, ha='center', fontweight='bold', color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

plt.title('R78: PINN Architecture for Navier-Stokes Inverse Problem — End to End',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('demos/r78_navier_stokes_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
