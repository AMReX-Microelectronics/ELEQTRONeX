# utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
COLUMNS = ['ID', 'Charge', 'Occupation', 'Potential', 'RelDiff', 'X', 'Y', 'Z']

def read_step_data(step, directory='.', z_offset=None):
    filename = os.path.join(directory, f"step{step:04d}.dat")
    data = pd.read_csv(filename,
                       delim_whitespace=True,
                       skiprows=1,
                       names=COLUMNS)
    if z_offset is not None:
        data['Z'] = data['Z'] - z_offset
    nm=1e-9
    data['X'] /=  nm
    data['Y'] /=  nm
    data['Z'] /=  nm
    return data

def read_multiple_steps(step_range, directory='.', z_offset=None, Vgs_values=None):
    all_data = []
    for step in step_range:
        df = read_step_data(step, directory, z_offset)
        df['Step'] = step
        if Vgs_values is not None:
            df['Vgs'] = Vgs_values[step]
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def plot_occupation_projection(data, plane='YZ', pltname="2D_projection.png", show_title=False, plttitle='',CNT_radius=0.782887):
    data = data.copy()
    for col in ['X', 'Y', 'Z', 'Occupation']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['X', 'Y', 'Z', 'Occupation'])

    if plane == 'YZ':
        x, y = data['Y'], data['Z']
        axis_xlabel='Length along nanotube / (nm)'
        axis_ylabel=r'Height along HfO$_2$ / (nm)'
    elif plane == 'YX':
        x, y = data['Y'], data['X']
        axis_xlabel='Length along nanotube / (nm)'
        axis_ylabel='Width / (nm)'
    else:
        raise ValueError("Plane must be 'YZ' or 'YX'")

    print('Xmin/max: ', np.min(x), np.max(x))
    print('Ymin/max: ', np.min(y), np.max(y))

    plt.figure(figsize=(26, 8))

    # Font sizes
    TickFontSize = 30
    AxisFontSize = 30

    sc = plt.scatter(
        x, y,
        c=data['Occupation'],
        cmap='Reds',
        s=100,
        alpha=1,
        edgecolors='white'
    )

    cbar = plt.colorbar(sc)
    cbar.set_label('Occupation', fontsize=AxisFontSize, rotation=0)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.yaxis.set_label_coords(1, 1.08)
    cbar.ax.tick_params(labelsize=TickFontSize, width=2)
    ax = plt.gca()  # get current axes
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', width=2, length=4, color='k')
    ax.tick_params(which='major', width=2, length=8, color='k')

    fontsize=30
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Increase the frame thickness
    plt.xlabel(axis_xlabel, fontsize=AxisFontSize)
    plt.ylabel(axis_ylabel, fontsize=AxisFontSize)

    plt.xticks(fontsize=TickFontSize)
    plt.yticks(fontsize=TickFontSize)
    if plane == 'YX':
        plt.axhline(y=CNT_radius, color = 'gray', linestyle = 'dashed', linewidth=2)
        plt.axhline(y=-CNT_radius, color = 'gray', linestyle = 'dashed', linewidth=2)

    if show_title is True:
        plt.title(plttitle, fontsize=AxisFontSize,pad=15)#, loc='left')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(pltname, bbox_inches = "tight")

    plt.show()



def plot_occupation_3d(data, color_by='Occupation', size=3, opacity=0.8):
    """
    Interactive 3D scatter plot using Graph Objects for full control.
    """

    # Ensure required columns are numeric
    data = data.copy()
    for col in ['X', 'Y', 'Z', color_by]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['X', 'Y', 'Z', color_by])

    TickFontSize = 16
    AxisFontSize = 22

    fig = go.Figure(data=go.Scatter3d(
        x=data['X'],
        y=data['Y'],
        z=data['Z'],
        mode='markers',
        marker=dict(
            size=size,
            color=data[color_by],
            colorscale='Reds',
            opacity=opacity,
            colorbar=dict(
                title=color_by,
                titlefont=dict(size=AxisFontSize, color="black"),
                tickfont=dict(size=22, color="black"),
                thickness=25,
                len=0.75
            )
        )
    ))
    fig.add_annotation(
        dict(
            showarrow=False,
            text="Total active charge = 162 e<br>V<sub>gs</sub> = -1.08 V",
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            align="left",
            font=dict(size=AxisFontSize, color="black")
        )
    )

    fig.add_annotation(
        dict(
            text="Height along<br>HfO\u2082 / (nm)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.1,  # adjust to move horizontally
            y=0.55,   # center vertically
            textangle=0,  # rotate to match Y axis
            font=dict(size=AxisFontSize, color="black"),
            xanchor="center",
            yanchor="middle"
        )
    )

    fig.add_annotation(
        dict(
            text="Length along<br>nanotube / (nm)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.27,  # adjust to move horizontally
            y=0.1,   # center vertically
            textangle=0,  # rotate to match Y axis
            font=dict(size=AxisFontSize, color="black"),
            xanchor="center",
            yanchor="middle"
        )
    )
    fig.add_annotation(
        dict(
            text="Width / (nm)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.93,  # adjust to move horizontally
            y=0.1,   # center vertically
            textangle=0,  # rotate to match Y axis
            font=dict(size=AxisFontSize, color="black"),
            xanchor="center",
            yanchor="middle"
        )
    )
    fig.update_layout(
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=2, b=2),
        scene=dict(
            xaxis=dict(
                title=dict(text='',
                           font=dict(size=AxisFontSize)),
                tickfont=dict(size=TickFontSize, color="black"),
                dtick=0.5,
                showgrid=True,
                showline=True,
                mirror=True,
                zeroline=False,
                ticks='outside'
            ),
            yaxis=dict(
                title=dict(text='',
                           font=dict(size=AxisFontSize)),
                tickfont=dict(size=TickFontSize, color="black"),
                dtick=10,
                showgrid=True,
                showline=True,
                mirror=True,
                zeroline=False,
                ticks='outside'
            ),
            zaxis=dict(
                title=dict(text='', font=dict(size=AxisFontSize)),
                tickfont=dict(size=TickFontSize, color="black"),
                dtick=1,
                showgrid=True,
                showline=True,
                mirror=True,
                zeroline=False,
                ticks='outside'
            )
        )
    )

    fig.show()


def custom_plot_single(fig, ax1, x_lst, y_lst, label_lst, xlim, ylim, label,pltname,
                       color=['k','r','b','g','o']*1,
                       linestyle=['solid','dashed','solid','dashed','solid','dashed'],
                       markertype=[None]*50,
                       fillstyle=['none']*50,
                       linewidth=50*[3],
                       markevery=[1]*50,
                       show_legend=True,
                       plt_outside=False, y_logscale = False):

    fig.patch.set_facecolor('white')
    ax1.patch.set_facecolor('white')

    for p in range(0, len(x_lst)):
        ax1.plot(x_lst[p], y_lst[p], color[p],
                 linewidth=linewidth[p],
                 #linestyle=linestyle[p],
                 marker=markertype[p],
                 fillstyle=fillstyle[p],
                 markevery=markevery[p],
                 markersize=12,
                 label=label_lst[p])
    ax1.tick_params(which='minor', width=2, length=4, color='k')
    ax1.tick_params(which='major', width=2, length=8, color='k')

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    if(show_legend):
        ax1.legend(prop={'size': 24},loc='best',
                   shadow=True,
                   frameon=False)#,bbox_to_anchor=(1.01,1.0))
    if (y_logscale):
        ax1.set_yscale('log')
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_xlabel(label[0], fontsize=30)#, fontdict=dict(weight='bold'))
    ax1.set_ylabel(label[1], fontsize=30)#, fontdict=dict(weight='bold'))
    fontsize=30
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)  # Increase the frame thickness
    if(plt_outside==False):
        plt.savefig(pltname, bbox_inches = "tight")
    return plt,pltname

def grayscale_gradient(n, min_brightness=0.1, max_brightness=0.6):
    """
    Generate n grayscale colors from lighter gray to black.
    min_brightness=0 → black, 1 → white.
    """
    return [
        f'#{int(255 * b):02x}{int(255 * b):02x}{int(255 * b):02x}'
        for b in np.linspace(max_brightness, min_brightness, n)
    ]

def plot_1d_profiles_all_steps(data, axis='Z', dz=0.1, z_min='auto', z_max='auto', CNT_radius=0.782887,
pltname="occupation_profile.png", show_legends=True, annotation=None):
    """
    Plots occupation profile as a function of `axis` for all steps in the data.
    Uses custom_plot_single for styling.
    """
    axis = axis.upper()
    if axis not in ['X','Y','Z']:
        raise ValueError("Axis must be 'X','Y','Z'")

    axis_label=None
    if axis == 'X':
        axis_label='Width / (nm)'
    if axis == 'Y':
        axis_label='Length along nanotube / (nm)'
    elif axis == 'Z':
        axis_label=r'Height along HfO$_2$ / (nm)'

    steps = sorted(data['Step'].unique())
    x_lst = []
    y_lst = []
    label_lst = []

    # Determine global z range if 'auto'
    if z_min == 'auto':
        z_min = data[axis].min()
    if z_max == 'auto':
        z_max = data[axis].max()

    bins = np.arange(z_min, z_max + dz, dz)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for step in steps:
        df_step = data[data['Step'] == step]
        vgs = df_step['Vgs'].iloc[0]
        coord = df_step[axis]
        charge_sum, _ = np.histogram(coord, bins=bins, weights=df_step['Occupation'])
        x_lst.append(bin_centers)
        y_lst.append(charge_sum)
        label_lst.append(f"$V_{{gs}}$ = {vgs:.2f} V")

    # Create figure and axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.18, 0.15, 0.8, 0.8])

    color = grayscale_gradient(len(x_lst))

    if axis == 'X':
        plt.axvline(x=CNT_radius, color = 'gray', linestyle = 'dashed', linewidth=2)
        plt.axvline(x=-CNT_radius, color = 'gray', linestyle = 'dashed', linewidth=2)

    if annotation:
        ax.text(**annotation)
    # Plot using your custom style
    custom_plot_single(
        fig, ax,
        x_lst, y_lst, label_lst,
        xlim=[z_min, z_max],
        ylim=[0, max([max(y) for y in y_lst]) * 1.1],
        label=[axis_label, "Total active charge / (e)"],
        pltname=pltname,
        color=grayscale_gradient(len(x_lst)),
        #color=['k']* len(x_lst),
        #color=['C' + str(i % 10) for i in range(len(x_lst))],
        linewidth=[3] * len(x_lst),
        linestyle=['solid'] * len(x_lst),
        markertype=['<','o', 's', '^', 'x'] * len(x_lst),
        fillstyle=['none'] * len(x_lst),
        markevery=[1] * len(x_lst),
        show_legend=show_legends,
        plt_outside=False,
        y_logscale=False
    )

    plt.show()



def plot_1d_occupation_profile(data, axis='Z', dz=1e-9, z_min='auto', z_max='auto'):
    axis = axis.upper()
    if axis not in ['X','Y','Z']:
        raise ValueError("Axis must be 'X','Y','Z'")

    coord = data[axis]
    if z_min == 'auto':
        z_min = coord.min()
    if z_max == 'auto':
        z_max = coord.max()
    bins = np.arange(z_min, z_max + dz, dz)
    charge_sum, _ = np.histogram(coord, bins=bins, weights=data['Occupation'])
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, charge_sum)
    plt.xlabel(f'{axis}')
    plt.ylabel('Total Charge')
    plt.title(f'1D Charge Profile along {axis}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
