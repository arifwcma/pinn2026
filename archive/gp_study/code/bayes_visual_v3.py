import matplotlib.pyplot as plt
import matplotlib.patches as patches

p_high = 0.30
p_low = 0.70
p_dark_given_high = 0.80
p_dark_given_low = 0.10

high_dark_area = p_high * p_dark_given_high
low_dark_area = p_low * p_dark_given_low
total_dark = high_dark_area + low_dark_area
p_high_given_dark = high_dark_area / total_dark
p_low_given_dark = low_dark_area / total_dark

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

high_light_h = 10 * (1 - p_dark_given_high)
high_dark_h = 10 * p_dark_given_high
ax.add_patch(patches.Rectangle((0, 0), 3, high_light_h, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((0, high_light_h), 3, high_dark_h, facecolor="#4A2810", edgecolor="black", linewidth=2))

low_light_h = 10 * (1 - p_dark_given_low)
low_dark_h = 10 * p_dark_given_low
ax.add_patch(patches.Rectangle((5, 0), 5, low_light_h, facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((5, low_light_h), 5, low_dark_h, facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.text(1.5, high_light_h + high_dark_h / 2, f"Dark\n{p_dark_given_high:.0%}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(1.5, high_light_h / 2, f"Light\n{1 - p_dark_given_high:.0%}", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#8B4513")

ax.text(7.5, low_light_h + low_dark_h / 2, f"Dark\n{p_dark_given_low:.0%}", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")
ax.text(7.5, low_light_h / 2, f"Light\n{1 - p_dark_given_low:.0%}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#8B4513")

ax.text(1.5, -1.2, f"p(dark|high) = {p_dark_given_high:.2f}", ha="center", fontsize=11, fontweight="bold", color="#4A2810")
ax.text(7.5, -1.2, f"p(dark|low) = {p_dark_given_low:.2f}", ha="center", fontsize=11, fontweight="bold", color="#4A2810")

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

high_width = p_high_given_dark * 10
low_width = p_low_given_dark * 10

ax.add_patch(patches.Rectangle((0, 0), high_width, 10, facecolor="#4A2810", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((high_width, 0), low_width, 10, facecolor="#8B6914", edgecolor="black", linewidth=2))

ax.text(high_width / 2, 5, f"High\nCarbon\n{p_high_given_dark:.0%}", ha="center", va="center",
        fontsize=15, fontweight="bold", color="white")
ax.text(high_width + low_width / 2, 5, f"Low\nCarbon\n{p_low_given_dark:.0%}", ha="center", va="center",
        fontsize=15, fontweight="bold", color="white")

ax.axvline(high_width, color="yellow", linewidth=3, linestyle="--")

ax.text(high_width / 2, -1.2, f"p(high|dark) = {p_high_given_dark:.2f}",
        ha="center", fontsize=11, fontweight="bold", color="#4A2810")
ax.text(high_width + low_width / 2, -1.2, f"p(low|dark) = {p_low_given_dark:.2f}",
        ha="center", fontsize=11, fontweight="bold", color="#8B6914")

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("bayes_theorem_visual.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to bayes_theorem_visual.png")
