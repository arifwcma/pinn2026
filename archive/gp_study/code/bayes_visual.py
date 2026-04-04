import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_title("All 100 Soils in Your Region", fontsize=13, fontweight="bold")

ax.add_patch(patches.Rectangle((0, 0), 3, 10, facecolor="#8B4513", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((3, 0), 7, 10, facecolor="#D2B48C", edgecolor="black", linewidth=2))

ax.text(1.5, 5, "High\nCarbon\n30%", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
ax.text(6.5, 5, "Low\nCarbon\n70%", ha="center", va="center", fontsize=14, fontweight="bold", color="black")

ax.text(1.5, -0.7, "p(A=high) = 0.30", ha="center", fontsize=10, color="#8B4513", fontweight="bold")
ax.text(6.5, -0.7, "p(A=low) = 0.70", ha="center", fontsize=10, color="#8B6914", fontweight="bold")
ax.text(5, 10.5, "PRIOR: What we believe before looking at color", ha="center", fontsize=11, style="italic")
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_title("Break Each Group by Color", fontsize=13, fontweight="bold")

ax.add_patch(patches.Rectangle((0, 0), 3, 2, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((0, 2), 3, 8, facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.add_patch(patches.Rectangle((3, 0), 7, 8, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((3, 8), 7, 2, facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.text(1.5, 6, "Dark\n80%", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(1.5, 1, "Light\n20%", ha="center", va="center", fontsize=11, fontweight="bold", color="#8B4513")

ax.text(6.5, 9, "Dark\n20%", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(6.5, 4, "Light\n80%", ha="center", va="center", fontsize=13, fontweight="bold", color="#8B4513")

ax.annotate("p(dark | high)\n= 0.80", xy=(0, 6), xytext=(-2.5, 7.5),
            fontsize=9, fontweight="bold", color="#4A2810",
            arrowprops=dict(arrowstyle="->", color="#4A2810"))
ax.annotate("p(dark | low)\n= 0.20", xy=(10, 9), xytext=(10.5, 7),
            fontsize=9, fontweight="bold", color="#4A2810",
            arrowprops=dict(arrowstyle="->", color="#4A2810"))

ax.text(5, 10.5, "LIKELIHOOD: How probable is dark brown for each group?", ha="center", fontsize=11, style="italic")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-3, 13)

ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_title("Only Keep Dark Brown Soils", fontsize=13, fontweight="bold")

total_dark_width = 10
high_dark_area = 0.30 * 0.80
low_dark_area = 0.70 * 0.20
total_dark = high_dark_area + low_dark_area
high_fraction = high_dark_area / total_dark
low_fraction = low_dark_area / total_dark

high_width = high_fraction * total_dark_width
low_width = low_fraction * total_dark_width

ax.add_patch(patches.Rectangle((0, 0), high_width, 10, facecolor="#4A2810", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((high_width, 0), low_width, 10, facecolor="#8B6914", edgecolor="black", linewidth=2))

ax.text(high_width / 2, 5, f"High\nCarbon\n{high_fraction:.0%}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(high_width + low_width / 2, 5, f"Low\nCarbon\n{low_fraction:.0%}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")

ax.axvline(high_width, color="yellow", linewidth=3, linestyle="--")

ax.text(5, -0.7, f"p(high | dark) = {high_fraction:.2f}     p(low | dark) = {low_fraction:.2f}",
        ha="center", fontsize=11, fontweight="bold", color="#4A2810")
ax.text(5, 10.5, "POSTERIOR: Among dark soils, what fraction is high carbon?", ha="center", fontsize=11, style="italic")
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(top=0.88)
fig.suptitle("Bayes' Theorem: Soil Carbon Example", fontsize=16, fontweight="bold", y=0.98)
plt.show()

print("=== The Numbers ===")
print(f"p(high carbon)              = 0.30   [PRIOR]")
print(f"p(low carbon)               = 0.70   [PRIOR]")
print(f"p(dark | high carbon)       = 0.80   [LIKELIHOOD]")
print(f"p(dark | low carbon)        = 0.20   [LIKELIHOOD]")
print(f"")
print(f"p(dark) = 0.30×0.80 + 0.70×0.20 = {total_dark:.2f}   [EVIDENCE]")
print(f"")
print(f"p(high carbon | dark) = (0.80 × 0.30) / {total_dark:.2f} = {high_fraction:.4f}   [POSTERIOR]")
print(f"p(low carbon | dark)  = (0.20 × 0.70) / {total_dark:.2f} = {low_fraction:.4f}   [POSTERIOR]")
