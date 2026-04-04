import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")

high_w = 3.0
low_w = 7.0

high_dark_h = 8.0
high_light_h = 2.0
low_dark_h = 1.0
low_light_h = 9.0

ax.add_patch(patches.Rectangle((0, 0), high_w, high_light_h,
             facecolor="#F5DEB3", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((0, high_light_h), high_w, high_dark_h,
             facecolor="#4A2810", edgecolor="black", linewidth=2))

ax.add_patch(patches.Rectangle((high_w, 0), low_w, low_light_h,
             facecolor="#FAEBD7", edgecolor="black", linewidth=2))
ax.add_patch(patches.Rectangle((high_w, low_light_h), low_w, low_dark_h,
             facecolor="#7B5B3A", edgecolor="black", linewidth=2))

ax.text(high_w / 2, high_light_h + high_dark_h / 2,
        "High Carbon\nDark Brown\n24 soils",
        ha="center", va="center", fontsize=14, fontweight="bold", color="white")

ax.text(high_w / 2, high_light_h / 2,
        "High Carbon\nLight\n6 soils",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#8B4513")

ax.text(high_w + low_w / 2, low_light_h + low_dark_h / 2,
        "Low Carbon, Dark Brown, 7 soils",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

ax.text(high_w + low_w / 2, low_light_h / 2,
        "Low Carbon\nLight\n63 soils",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#8B4513")

ax.annotate("", xy=(high_w, -0.8), xytext=(0, -0.8),
            arrowprops=dict(arrowstyle="<->", color="#8B4513", lw=2))
ax.text(high_w / 2, -1.3, "High Carbon: 30 soils", ha="center", fontsize=12, fontweight="bold", color="#8B4513")

ax.annotate("", xy=(10, -0.8), xytext=(high_w, -0.8),
            arrowprops=dict(arrowstyle="<->", color="#8B6914", lw=2))
ax.text(high_w + low_w / 2, -1.3, "Low Carbon: 70 soils", ha="center", fontsize=12, fontweight="bold", color="#8B6914")

ax.annotate("", xy=(-0.8, high_light_h), xytext=(-0.8, 10),
            arrowprops=dict(arrowstyle="<->", color="#4A2810", lw=2))
ax.text(-1.8, high_light_h + high_dark_h / 2, "Dark\n80%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#4A2810", rotation=0)

ax.annotate("", xy=(-0.8, 0), xytext=(-0.8, high_light_h),
            arrowprops=dict(arrowstyle="<->", color="#D2B48C", lw=2))
ax.text(-1.8, high_light_h / 2, "Light\n20%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#8B6914", rotation=0)

ax.annotate("", xy=(10.8, low_light_h), xytext=(10.8, 10),
            arrowprops=dict(arrowstyle="<->", color="#4A2810", lw=2))
ax.text(11.8, low_light_h + low_dark_h / 2, "Dark\n10%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#4A2810", rotation=0)

ax.annotate("", xy=(10.8, 0), xytext=(10.8, low_light_h),
            arrowprops=dict(arrowstyle="<->", color="#D2B48C", lw=2))
ax.text(11.8, low_light_h / 2, "Light\n90%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#8B6914", rotation=0)

ax.set_xlim(-3, 13)
ax.set_ylim(-2, 11)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("All 100 Soils: 4 Groups", fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("bayes_four_groups.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to bayes_four_groups.png")
