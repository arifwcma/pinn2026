import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Bayes' Theorem: Soil Carbon Example", fontsize=18, fontweight="bold", y=0.99)

ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(-2, 12)
ax.set_aspect("equal")
ax.set_title("PRIOR: Before looking at color\nAll 100 soils in your region", fontsize=12, fontweight="bold", pad=15)

ax.add_patch(patches.Rectangle((0, 0), 3, 10, facecolor="#8B4513", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((3, 0), 7, 10, facecolor="#D2B48C", edgecolor="black", linewidth=2))

ax.text(1.5, 5, "High\nCarbon\n30%", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
ax.text(6.5, 5, "Low\nCarbon\n70%", ha="center", va="center", fontsize=15, fontweight="bold", color="black")

ax.text(1.5, -1.2, "p(high) = 0.30", ha="center", fontsize=11, color="#8B4513", fontweight="bold")
ax.text(6.5, -1.2, "p(low) = 0.70", ha="center", fontsize=11, color="#8B6914", fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax = axes[1]
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 12)
ax.set_aspect("equal")
ax.set_title("LIKELIHOOD: How probable is\ndark brown for each group?", fontsize=12, fontweight="bold", pad=15)

ax.add_patch(patches.Rectangle((0, 0), 3, 2, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((0, 2), 3, 8, facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.add_patch(patches.Rectangle((5, 0), 5, 8, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((5, 8), 5, 2, facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.text(1.5, 6, "Dark\n80%", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
ax.text(1.5, 1, "Light\n20%", ha="center", va="center", fontsize=11, fontweight="bold", color="#8B4513")

ax.text(7.5, 9, "Dark\n20%", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
ax.text(7.5, 4, "Light\n80%", ha="center", va="center", fontsize=14, fontweight="bold", color="#8B4513")

ax.text(1.5, -1.2, "p(dark|high) = 0.80", ha="center", fontsize=11, fontweight="bold", color="#4A2810")
ax.text(7.5, -1.2, "p(dark|low) = 0.20", ha="center", fontsize=11, fontweight="bold", color="#4A2810")

ax.text(1.5, 10.7, "High Carbon", ha="center", fontsize=11, fontweight="bold", color="#8B4513")
ax.text(7.5, 10.7, "Low Carbon", ha="center", fontsize=11, fontweight="bold", color="#8B6914")

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-2, 12)
ax.set_aspect("equal")
ax.set_title("POSTERIOR: Among dark soils only,\nwhat fraction is high carbon?", fontsize=12, fontweight="bold", pad=15)

high_dark_area = 0.30 * 0.80
low_dark_area = 0.70 * 0.20
total_dark = high_dark_area + low_dark_area
high_fraction = high_dark_area / total_dark
low_fraction = low_dark_area / total_dark

high_width = high_fraction * 10
low_width = low_fraction * 10

ax.add_patch(patches.Rectangle((0, 0), high_width, 10, facecolor="#4A2810", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((high_width, 0), low_width, 10, facecolor="#8B6914", edgecolor="black", linewidth=2))

ax.text(high_width / 2, 5, f"High\nCarbon\n{high_fraction:.0%}", ha="center", va="center",
        fontsize=15, fontweight="bold", color="white")
ax.text(high_width + low_width / 2, 5, f"Low\nCarbon\n{low_fraction:.0%}", ha="center", va="center",
        fontsize=15, fontweight="bold", color="white")

ax.axvline(high_width, color="yellow", linewidth=3, linestyle="--")

ax.text(high_width / 2, -1.2, f"p(high|dark) = {high_fraction:.2f}",
        ha="center", fontsize=11, fontweight="bold", color="#4A2810")
ax.text(high_width + low_width / 2, -1.2, f"p(low|dark) = {low_fraction:.2f}",
        ha="center", fontsize=11, fontweight="bold", color="#8B6914")

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("bayes_theorem_visual.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to bayes_theorem_visual.png")
