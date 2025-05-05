import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Combined data
models = [
    'fixed_sd_double',
    'fixed_mlp_double',
    'fixed_sd_same',
    'fixed_mlp_same',
    'fixed_sd_half',
    'fixed_mlp_half'
]
relative_speed      = [9.53375, 8.87085, 5.266185, 4.39185, 2.3757, 3.72045]
trajectory_distance = [7.87065, 8.67585, 3.9701, 4.08965, 1.831, 2.0721]
unique_per_100      = [31.855, 59.54, 11.775, 21.685, 6.26, 12.57]

# Color themes
colors1 = ['skyblue'    if 'sd' in m else 'salmon'      for m in models]
colors2 = ['lightgreen' if 'sd' in m else 'mediumorchid' for m in models]
colors3 = ['gold'       if 'sd' in m else 'slategray'   for m in models]

# Legend patches
patch1 = [mpatches.Patch(color='skyblue',    label='fixed sd'),
          mpatches.Patch(color='salmon',     label='fixed MLP')]
patch2 = [mpatches.Patch(color='lightgreen', label='fixed sd'),
          mpatches.Patch(color='mediumorchid', label='fixed MLP')]
patch3 = [mpatches.Patch(color='gold',       label='fixed sd'),
          mpatches.Patch(color='slategray',  label='fixed MLP')]

# 1) Relative Speed (6×4 inches)
plt.figure(figsize=(7,4))
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.bar(models, relative_speed, color=colors1, alpha=0.82, zorder=3)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Relative speed')
plt.title('Relative Speed by Model Version')
plt.legend(handles=patch1)
plt.tight_layout()
plt.show()

# 2) Total Trajectory Distance (6×4 inches)
plt.figure(figsize=(7,4))
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.bar(models, trajectory_distance, color=colors2, alpha=0.82, zorder=3)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total trajectory distance')
plt.title('Total Trajectory Distance by Model Version')
plt.legend(handles=patch2)
plt.tight_layout()
plt.show()

# 3) Unique per 100 (6×4 inches)
plt.figure(figsize=(7,4))
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.bar(models, unique_per_100, color=colors3, alpha=0.82, zorder=3)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Unique per 100')
plt.title('Unique per 100 by Model Version')
plt.legend(handles=patch3)
plt.tight_layout()
plt.show()
