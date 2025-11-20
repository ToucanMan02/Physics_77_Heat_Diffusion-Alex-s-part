import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

"""
Create a clean comparison table visualization for the stability report
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# ============================================================================
# LEFT PLOT: Method Comparison Table
# ============================================================================
ax1.axis('off')

# Table data
methods = ['Explicit\n(Forward Euler)', 'Crank-Nicolson\n(ADI)']
criteria = [
    'Stability Condition',
    'Time Step Limit',
    'Accuracy (Time)',
    'Accuracy (Space)',
    'Cost per Step',
    'Implementation',
    'Memory Usage',
    'Best for High α',
    'Best for Long Time'
]

data = [
    ['dt < dx²/(4α)', 'Unconditional'],
    ['Severe\n(~10⁻⁵)', 'None\n(can use ~10⁻²)'],
    ['First-order O(dt)', 'Second-order O(dt²)'],
    ['Second-order O(dx²)', 'Second-order O(dx²)'],
    ['Low', 'High'],
    ['Simple', 'Complex'],
    ['Low', 'High'],
    ['❌ No', '✓ Yes'],
    ['❌ No', '✓ Yes']
]

# Colors for cells
colors_explicit = ['#ffcccc', '#ffcccc', '#ffffcc', '#ccffcc', '#ccffcc', '#ccffcc', '#ccffcc', '#ffcccc', '#ffcccc']
colors_cn = ['#ccffcc', '#ccffcc', '#ccffcc', '#ccffcc', '#ffffcc', '#ffffcc', '#ffffcc', '#ccffcc', '#ccffcc']

# Create table
n_rows = len(criteria)
n_cols = 3  # criterion + 2 methods

cell_height = 1
cell_width_crit = 3.5
cell_width_method = 2.5

# Draw header
header_y = n_rows * cell_height + 1
ax1.add_patch(Rectangle((0, header_y), cell_width_crit, 1, 
                         facecolor='#4CAF50', edgecolor='black', linewidth=2))
ax1.text(cell_width_crit/2, header_y + 0.5, 'Criterion', 
         ha='center', va='center', fontsize=14, fontweight='bold', color='white')

for i, method in enumerate(methods):
    x = cell_width_crit + i * cell_width_method
    ax1.add_patch(Rectangle((x, header_y), cell_width_method, 1,
                           facecolor='#2196F3', edgecolor='black', linewidth=2))
    ax1.text(x + cell_width_method/2, header_y + 0.5, method,
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Draw data rows
for i, (crit, row_data) in enumerate(zip(criteria, data)):
    y = (n_rows - i - 1) * cell_height
    
    # Criterion cell
    ax1.add_patch(Rectangle((0, y), cell_width_crit, cell_height,
                           facecolor='#e3f2fd', edgecolor='black', linewidth=1))
    ax1.text(cell_width_crit/2, y + cell_height/2, crit,
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Method cells
    for j, (value, color) in enumerate(zip(row_data, [colors_explicit[i], colors_cn[i]])):
        x = cell_width_crit + j * cell_width_method
        ax1.add_patch(Rectangle((x, y), cell_width_method, cell_height,
                               facecolor=color, edgecolor='black', linewidth=1))
        ax1.text(x + cell_width_method/2, y + cell_height/2, value,
                ha='center', va='center', fontsize=10)

ax1.set_xlim(0, cell_width_crit + 2 * cell_width_method)
ax1.set_ylim(0, header_y + 1.5)
ax1.set_title('Method Comparison: Explicit vs Crank-Nicolson', 
              fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_y = -1
ax1.text(1, legend_y, 'Color Code:', fontsize=10, fontweight='bold')
ax1.add_patch(Rectangle((2.5, legend_y-0.2), 0.5, 0.4, facecolor='#ccffcc'))
ax1.text(3.2, legend_y, 'Advantage', fontsize=9, va='center')
ax1.add_patch(Rectangle((4.2, legend_y-0.2), 0.5, 0.4, facecolor='#ffffcc'))
ax1.text(4.9, legend_y, 'Neutral', fontsize=9, va='center')
ax1.add_patch(Rectangle((5.9, legend_y-0.2), 0.5, 0.4, facecolor='#ffcccc'))
ax1.text(6.6, legend_y, 'Disadvantage', fontsize=9, va='center')

# ============================================================================
# RIGHT PLOT: Project-Specific Analysis
# ============================================================================
ax2.axis('off')

# Project files analysis
project_files = [
    {
        'name': 'heatEquation2D.py',
        'method': 'Explicit',
        'alpha': 110,
        'dt_required': '5.68×10⁻⁶',
        'status': 'Very Small dt Required',
        'color': '#ff9999'
    },
    {
        'name': 'explicit_planet_impact_luke_ver_1.py',
        'method': 'Explicit',
        'alpha': 500,
        'dt_required': '5.00×10⁻⁷',
        'status': 'Extremely Small dt',
        'color': '#ff6666'
    },
    {
        'name': 'smooth_heat.py',
        'method': 'Crank-Nicolson',
        'alpha': 0.1,
        'dt_used': '1.00×10⁻⁴',
        'status': 'Stable & Efficient',
        'color': '#99ff99'
    },
    {
        'name': 'implicit/implicit.py',
        'method': 'Crank-Nicolson',
        'alpha': 0.8,
        'dt_used': '1.00×10⁻⁴',
        'status': 'Stable & Efficient',
        'color': '#99ff99'
    }
]

# Title
ax2.text(0.5, 0.95, 'Project Files Stability Analysis', 
         ha='center', fontsize=16, fontweight='bold',
         transform=ax2.transAxes)

# Draw file analysis boxes
y_start = 0.85
box_height = 0.18
box_spacing = 0.02

for i, file_info in enumerate(project_files):
    y = y_start - i * (box_height + box_spacing)
    
    # Background box
    rect = mpatches.FancyBboxPatch((0.05, y), 0.9, box_height,
                                   boxstyle="round,pad=0.01",
                                   facecolor=file_info['color'],
                                   edgecolor='black', linewidth=2,
                                   transform=ax2.transAxes)
    ax2.add_patch(rect)
    
    # File name
    ax2.text(0.08, y + box_height - 0.03, file_info['name'],
            fontsize=12, fontweight='bold', transform=ax2.transAxes)
    
    # Method
    ax2.text(0.08, y + box_height/2, f"Method: {file_info['method']}",
            fontsize=10, transform=ax2.transAxes)
    
    # Alpha value
    ax2.text(0.08, y + 0.03, f"α = {file_info['alpha']}",
            fontsize=10, fontweight='bold', transform=ax2.transAxes)
    
    # Time step info
    if 'dt_required' in file_info:
        dt_text = f"dt required: {file_info['dt_required']}"
    else:
        dt_text = f"dt used: {file_info['dt_used']}"
    ax2.text(0.5, y + box_height/2, dt_text,
            fontsize=10, transform=ax2.transAxes)
    
    # Status
    ax2.text(0.92, y + box_height/2, file_info['status'],
            fontsize=9, fontweight='bold', ha='right',
            transform=ax2.transAxes)

# Key insights box
insights_y = 0.05
insight_box = mpatches.FancyBboxPatch((0.05, insights_y), 0.9, 0.15,
                                      boxstyle="round,pad=0.01",
                                      facecolor='#e3f2fd',
                                      edgecolor='#2196F3', linewidth=2,
                                      transform=ax2.transAxes)
ax2.add_patch(insight_box)

ax2.text(0.5, insights_y + 0.115, 'Key Insights', 
         ha='center', fontsize=12, fontweight='bold',
         transform=ax2.transAxes)

insights = [
    "• Explicit methods with high α require extremely small dt (up to 1000× smaller)",
    "• Crank-Nicolson methods can use ~200× larger dt while maintaining stability",
    "• For high diffusivity problems (α > 100), Crank-Nicolson is strongly recommended"
]

for i, insight in enumerate(insights):
    ax2.text(0.08, insights_y + 0.08 - i*0.025, insight,
            fontsize=9, transform=ax2.transAxes)

# ============================================================================
# Overall title
# ============================================================================
fig.suptitle('Stability Analysis: Heat Diffusion Methods Comparison', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/jasonfan/Downloads/Physics_77_Heat_Diffusion/stability_comparison_table.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Comparison table saved as 'stability_comparison_table.png'")
plt.show()


