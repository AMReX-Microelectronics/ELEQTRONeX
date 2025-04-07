from scipy.optimize import fsolve
from numpy.linalg import inv
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
def custom_plot_single(fig, ax1, 
                       x_lst, y_lst, label_lst, xlim, ylim, label,pltname,
                       color=['k','r','b','g','o','c','m','y'],
                       linestyle=['solid','dashed','solid','dashed','solid','dashed'],
                       markertype=[None,None,'o','^','o','^'],
                       fillstyle=['none','none','none','none','full','full'],
                       linewidth=20*[3],
                       markevery=[450,30,50,40,56,72,63,95],
                       show_legend=True,
                       plt_outside=False, y_logscale = False, fontsize=30, spine_edgecolor='k'):
    
    fig.patch.set_facecolor('white')
    ax1.patch.set_facecolor('white')
    #color[len(x_lst)-1] = 'gray'
    for p in range(0, len(x_lst)):
        ax1.plot(x_lst[p], y_lst[p], color[p],
                 linewidth=linewidth[p],
                 linestyle=linestyle[p],
                 marker=markertype[p],
                 fillstyle=fillstyle[p],
                 markevery=markevery[p],
                 markerfacecolor='None', 
                 markersize=14,
                 label=label_lst[p])
    if(show_legend):
        ax1.legend(prop={'size': 30},loc='best',shadow=True, frameon=False)
        
    ax1.tick_params(which='minor', width=2, length=4, color=spine_edgecolor)
    ax1.tick_params(which='major', width=2, length=8, color=spine_edgecolor)

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if (y_logscale):
        ax1.set_yscale('log')        
        ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    else:
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_xlabel(label[0], fontsize=fontsize)#, fontdict=dict(weight='bold'))
    ax1.set_ylabel(label[1], fontsize=fontsize)#, fontdict=dict(weight='bold'))
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)  # Increase the frame thickness     
        spine.set_edgecolor(spine_edgecolor)  # Set the spine edge color to red    
        
    if(plt_outside==False):
        plt.savefig(pltname, bbox_inches = "tight")
        
    return plt,pltname

##### # Function to find the x-value for given y-value
def find_x_for_y(target_y, x_range, f):
    # Define the equation f(x) - target_y = 0
    func = lambda x: f(x) - target_y
    # Use fsolve to find the root, with an initial guess
    x_root, = fsolve(func, np.mean(x_range))
    return x_root

def compute_barrier(U, Vgs, fig, ax, Ef=-0.2, a=5, b=8, Print=True, Eg=0.4535):
    num_steps = np.size(Vgs)
    Ev = -Eg/2. + U
    Ec = Eg/2 + U
    #print(Ev[0][0])
    barrier = np.zeros(num_steps, dtype=float)
    phonon_jump = np.zeros(num_steps, dtype=float)
    
    for j in range(num_steps):
        mid = int(np.shape(Ev)[-1]/2)
        barrier[j] = Ef - Ev[j][mid]
        phonon_jump[j] = Ec[j][mid] - Ef
           # print(Ev[j][0])

    kT_eV = 0.0257025 #eV 1.38e-23*298/1.6e-19    

    f = interp1d(Vgs, barrier, kind='linear', fill_value='extrapolate')      
    #print(factor1*kT_eV, f(factor1*kT_eV))
    # Find x-values where f(x) equals kT_eV and 4*kT_eV
    x_akT_eV = find_x_for_y(a*kT_eV, Vgs, f)
    x_bkT_eV = find_x_for_y(b*kT_eV, Vgs, f)     
    g = interp1d(Vgs, phonon_jump, kind='linear', fill_value='extrapolate')
    x_dot2eV = find_x_for_y(0.2, Vgs, g)

    print('Vgs where barrier is ', a, 'kT:', x_akT_eV)
    print('Vgs where barrier is ', b, 'kT:', x_bkT_eV)
    print('Vgs where phonon-tunneling barrier is 0.2 eV:', x_dot2eV)
    
    import matplotlib.pyplot as plt

    if(Print):           
        xmin = Vgs[-1]- 0.1
        xmax = Vgs[0] + 0.1
        ymin = -0.1
        ymax = 0.6 #max range
        plt,pltname = custom_plot_single(fig, ax,
                                         [Vgs],
                                         [barrier],
                                         [r'Transport barrier']                                          
                                         #[Vgs, Vgs],
                                         #[barrier, phonon_jump],
                                         #[r'Transport barrier', r'Phonon-tunneling barrier'] 
                                         + [None for s in range(num_steps)],
                                         [xmin,xmax],
                                         [ymin,ymax],
                                         [r'$V_{gs}$ / (V)',
                                          r'Barrier / (eV)'],
                                         'Barrier.png',
                                         color = ['k','r'],
                                         linestyle=['solid','solid'],
                                         markertype=[None,None],
                                         markevery=[1,1,1,1,1,1,1,1,1,1,1,1],
                                         linewidth=[1,1,1,1,1,1,1,1,1,1,1,3,3],
                                         fillstyle=['none','none','none','none','none','none','none','none','none','none','none','none','none','none'],                                     
                                         show_legend=False,
                                         plt_outside=True, y_logscale=False) 
        

    
        ax.plot([xmin,x_akT_eV], [a*kT_eV,a*kT_eV], color='k', linestyle='--')     
        ax.plot([xmin,x_bkT_eV], [b*kT_eV,b*kT_eV], color='k', linestyle='--')     
    
        ax.plot([x_akT_eV,x_akT_eV], [ymin,a*kT_eV], color='k', linestyle='--')     
        ax.plot([x_bkT_eV,x_bkT_eV], [ymin,b*kT_eV], color='k', linestyle='--') 

        #ax.plot([x_dot2eV,x_dot2eV], [ymin,0.2], color='r', linestyle='--')     
        #ax.plot([xmin,x_dot2eV], [0.2,0.2], color='r', linestyle='--')     
  
        #plt.savefig(pltname, bbox_inches = "tight", dpi='figure', format='png', pad_inches=0.1, facecolor='auto', edgecolor='auto')
        
    return x_akT_eV, x_bkT_eV, x_dot2eV

def log_interp(Vgs, Ids, at_Vgs):

    # Log-transform the Ids
    log_Ids = np.log(Ids)
    
    # Create a linear interpolation function on the log-transformed data
    log_interp = interp1d(Vgs, log_Ids, kind='linear')
    
    # Interpolate to find the log of the interpolated value at at_Vgs
    log_interpolated_value = log_interp(at_Vgs)
    
    # Exponentiate to get back to the original scale
    interpolated_value = np.exp(log_interpolated_value)
    
    return interpolated_value


def compute_SS(Vgs, G, at_numer_Vgs, at_denom_Vgs, Print=True):
    Vds=0.1 #[V]
    G_0 = 7.748091729e-5 #[S]
    Scaling_Factor = Vds*G_0
    
    Ids = Scaling_Factor*G
    #print('\nVgs:', Vgs)
    #print('Ids:', Ids)
    dVg = at_denom_Vgs - at_numer_Vgs
    dlogI = np.log10(log_interp(Vgs, Ids, at_numer_Vgs)/log_interp(Vgs, Ids, at_denom_Vgs))
    SS = dVg*1e3/dlogI #mV/decade
    print('\tSS (mV/decade):', round(SS,1))

    return SS