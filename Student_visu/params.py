import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

colors3 = ['#2ba77b', '#e9a001', '#2274b2']

def plot_model_comparison(teacher_params, student1_params, student2_params,
                          base_radius=0.5, inset_position=None, zoom_factor=100, save_path=None):

    font_size = 15
    font = {'size': font_size}
    plt.rc('font', **font)
    plt.rcParams['axes.linewidth'] = 1.50

    # Calculate the multiplying factors based on parameters
    total_params = teacher_params + student1_params + student2_params
    teacher_factor = teacher_params / total_params
    student1_factor = student1_params / total_params
    student2_factor = student2_params / total_params

    # Normalize radii
    teacher_radius = base_radius * teacher_factor
    student1_radius = base_radius * student1_factor
    student2_radius = base_radius * student2_factor

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(6, 4), dpi=1200)

    # Determine the bottom baseline for the circles
    baseline_y = 0.1 * base_radius  # Margin around the plot

    # Calculate positions to minimize white space and ensure visibility of all three circles
    distance = 0.1 * base_radius  # Small gap between circles
    teacher_center = (teacher_radius + distance, baseline_y + teacher_radius)
    student1_center = (teacher_center[0] + teacher_radius + student1_radius + distance,
                       baseline_y + student1_radius)
    student2_center = (student1_center[0] + student1_radius + student2_radius + distance,
                       baseline_y + student2_radius)

    # Add circles for teacher and students
    teacher_circle = plt.Circle(teacher_center, teacher_radius, edgecolor='none',
                                facecolor=colors3[0], label='Teacher', alpha=0.5)
    student1_circle = plt.Circle(student1_center, student1_radius, edgecolor='none',
                                 facecolor=colors3[1], label='Student 1', alpha=0.5)
    student2_circle = plt.Circle(student2_center, student2_radius, edgecolor='none',
                                 facecolor=colors3[2], label='Student 2', alpha=0.5)

    # Add circles to the main plot
    ax.add_artist(teacher_circle)
    ax.add_artist(student1_circle)
    ax.add_artist(student2_circle)

    # Setting the limits of the main plot
    max_radius = max(teacher_radius, student1_radius, student2_radius)
    ax.set_xlim(0, student2_center[0] + student2_radius + 0.1 * base_radius)
    ax.set_ylim(0, baseline_y + 2 * max_radius + 0.1 * base_radius)
    ax.set_aspect('equal', 'box')

    # Remove axis from the main plot
    ax.axis('off')

    # Adding legend
    plt.legend([teacher_circle, student1_circle, student2_circle],
               ['Teacher', 'Student 1', 'Student 2'])

    # Create a zoomed inset axes with a border
    if inset_position is not None:
        # inset_position should be [x0, y0, width, height] in axes coordinates (0 to 1)
        axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='right', borderpad=0,
                                  bbox_to_anchor=inset_position,
                                  bbox_transform=ax.transAxes)
    else:
        # Default position if inset_position is not provided
        axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='center right', borderpad=1)

    # Add circles to the inset plot
    teacher_circle_ins = plt.Circle(teacher_center, teacher_radius, edgecolor='none',
                                    facecolor=colors3[0], alpha=0.5)
    student1_circle_ins = plt.Circle(student1_center, student1_radius, edgecolor='none',
                                     facecolor=colors3[1], alpha=0.5)
    student2_circle_ins = plt.Circle(student2_center, student2_radius, edgecolor='none',
                                     facecolor=colors3[2], alpha=0.5)

    axins.add_patch(teacher_circle_ins)
    axins.add_patch(student1_circle_ins)
    axins.add_patch(student2_circle_ins)

    # Set the limits of the inset plot to focus on the Student 2 circle
    x1 = student2_center[0] - 2 * student2_radius
    x2 = student2_center[0] + 2 * student2_radius
    y1 = student2_center[1] - 2 * student2_radius
    y2 = student2_center[1] + 2 * student2_radius

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_aspect('equal', 'box')

    # Remove tick labels from the inset plot
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    # Add a border around the inset axes
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Indicate the zoomed area on the main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.savefig(save_path + '.png', format='png', dpi=1200, bbox_inches='tight')

    # Display the plot
    plt.show()

# Example usage
T_pram = 23512130
s1_pram = 11171266
s2_pram = 5682

# Specify the inset position [x0, y0, width, height] in axes coordinates (0 to 1)
inset_position = [0.72, 0.1, 0.25, 0.25]  # Adjust these values as needed

# Call the function with the desired zoom factor
plot_model_comparison(T_pram, s1_pram, s2_pram, inset_position=inset_position, zoom_factor=100, save_path="params")
