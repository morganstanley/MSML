import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import ticker
from utils import *
from sim_process import BB_get_drift
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_particle(dt, 
                  drift_test, 
                  drift_MLP, 
                  epoch, 
                  plot_savepath, 
                  simulation,
                  paths=False,
                  train=False, 
                  plot_particles=5, 
                  train_split_t=None,
                  drift_fore=None,
                  irreg_t=None,
                  test=False,
                  head="None"
                 ):
    ts = dt*np.array(range(drift_test.shape[-2]))
    fig, ax = plt.subplots(figsize=(10, drift_test.shape[-1]*5), nrows=drift_test.shape[-1], ncols=2)
    plot_particles = plot_particles if drift_test.shape[0] > 3 else np.min(plot_particles,drift_test.shape[0])
    for dim in range(drift_test.shape[-1]):
        for data_type in range(len(["Real", "Generated"])):
            for particle in range(plot_particles):
                if drift_test.shape[-1] == 1:
                    ax_temp = ax[data_type]
                else: 
                    ax_temp = ax[dim][data_type]
                if data_type == 0:
                    if len(drift_test.shape) > 3:
                        sns.lineplot(x=ts, y=drift_test[0,particle,:,dim], ax=ax_temp)
                    else:
                        colors = plt.cm.inferno(np.linspace(0,1,plot_particles))
                        sns.lineplot(x=ts, y=drift_test[particle,:,dim], ax=ax_temp, color = colors[particle], alpha=0.3)
                    
                    if train is False and train_split_t is not None:
                        if len(drift_test.shape) > 3:
                            sns.lineplot(x=ts[train_split_t-1:], y=drift_fore[0,particle,:,dim], ax=ax_temp, color="black")
                        else:
                            sns.lineplot(x=ts[train_split_t-1:], y=drift_fore[particle,:,dim], ax=ax_temp, color="black")
                else:
                    if len(drift_MLP.shape) > 3:
                        sns.lineplot(x=ts, y=drift_MLP[0,particle,:,dim], ax=ax_temp)
                    else:
                        colors = plt.cm.inferno(np.linspace(0,1,plot_particles))
                        sns.lineplot(x=ts, y=drift_MLP[particle,:,dim], ax=ax_temp, color = colors[particle], alpha=0.3)
                    if irreg_t is not None:
                        sns.scatterplot(x=ts[irreg_t], y=drift_test[particle,irreg_t,dim], ax=ax_temp, color="grey", alpha=0.3)
                        
            if train is False and train_split_t is not None:
                ax_temp.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
            ax_temp.set_title("Dim: {}, First {} Particles, {} Data".format(dim, plot_particles, 
                                                                            ["Empirical", "Gen"][data_type]))
    if test is False and simulation is True and paths is True:
        plt.savefig(plot_savepath + "Particle_paths_{}_Epoch_{}.pdf".format(("Train" if train else "Val"),
                                                                        epoch+1),bbox_inches='tight', format="pdf")
    elif test is False and simulation is True and paths is False:
        plt.savefig(plot_savepath + "Particle_drift_{}_Epoch_{}.pdf".format(("Train" if train else "Val"),
                                                                       epoch+1),bbox_inches='tight', format="pdf") 
    elif test is False and simulation is False and paths is False: 
        plt.savefig(plot_savepath + "Particle_paths_{}_Epoch_{}.pdf".format(("Train" if train else "Val"),
                                                                            epoch+1),bbox_inches='tight', format="pdf")
    elif test is True and simulation is True and paths is True:
        plt.savefig(plot_savepath + "{}_test_Particle_paths.pdf".format(head),bbox_inches='tight', format="pdf")
    elif test is False and simulation is True and paths is False:
        plt.savefig(plot_savepath + "{}_test_Particle_drift.pdf".format(head),bbox_inches='tight', format="pdf") 
    elif test is False and simulation is False and paths is False: 
        plt.savefig(plot_savepath + "{}_test_Particle_paths.pdf".format(head),bbox_inches='tight', format="pdf")
        
    plt.close("all")
                
def plot_gen_scatter(gen_path_simu, xs_val, epoch, plot_savepath, train=False, test=False):
    t_list = list(range(0,gen_path_simu.shape[1], int(gen_path_simu.shape[1]/5))) + [gen_path_simu.shape[1] - 1]
    fig, ax = plt.subplots(figsize=((len(t_list))*5, 10), nrows=2, ncols=len(t_list))
    if xs_val.shape[-1] == 3:
        fig = plt.figure(figsize=((len(t_list))*5, 10))
    for i,t in enumerate(t_list):
        if xs_val.shape[-1] == 3:
            x, y, z = gen_path_simu[:,t,:].T
            ax_temp = fig.add_subplot(2, len(t_list), i+1, projection='3d')
            ax_temp.scatter(x, y, z)
            
            ax_temp.view_init(azim=-60, elev=9)
            ax_temp.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_temp.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_temp.zaxis.set_major_locator(ticker.MultipleLocator(1))
            
            x, y, z = xs_val[:,t,:].T
            ax_temp = fig.add_subplot(2, len(t_list), i+1+len(t_list), projection='3d')
            ax_temp.scatter(x, y, z)
            
            ax_temp.view_init(azim=-60, elev=9)
            ax_temp.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_temp.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_temp.zaxis.set_major_locator(ticker.MultipleLocator(1))
            
        else: 
            ax_temp = ax[0][i]
            ax_temp.scatter(xs_val[:,t,0], xs_val[:,t,1], label="True")
            ax_temp.scatter(gen_path_simu[:,t,0], gen_path_simu[:,t,1], label="generated")
            ax_temp.legend()
        #ax_temp.title("T = {}".format(np.round(t)))
    if test == False:
        plt.savefig(plot_savepath + "Generated_samples_{}_Epoch_{}.pdf".format(("Train" if train else "Val"),
                                                                                epoch+1),bbox_inches='tight', format="pdf")
    elif test == True:
        plt.savefig(plot_savepath + "Test_Generated_samples.pdf",bbox_inches='tight', format="pdf")
        
    plt.close("all")



def plot_drift(dt, 
               drift_test,
               drift_MLP, 
               epoch, 
               plot_savepath,
               simulation, 
               train=False, 
               train_split_t=None,
               drift_fore = None, 
               test=False,
               head="None",
              ):
    ts = dt*np.array(range(drift_test.shape[-2]))
    plt.figure(figsize=(10,drift_test.shape[-1]*3))
    for dim in range(drift_test.shape[-1]):
        ax = plt.subplot(drift_test.shape[-1],1,dim+1)
        if len(drift_test.shape) > 3:
            test_mean = drift_test.mean(0).mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0).mean(0)[:,dim]
            if drift_fore and train_split_t:
                fore_mean = drift_fore.mean(0).mean(0)[:, dim]
            
            test_std = drift_test.std(1).mean(0)[:,dim]
            gen_std = drift_MLP.std(1).mean(0)[:,dim]
            if drift_fore and train_split_t:
                fore_std = drift_fore.std(1).mean(0)[:,dim]
            
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t and train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
        else:
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            if train_split_t and train_split_t:
                fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            if train_split_t and train_split_t:
                fore_std = drift_fore.std(0)[:,dim]
            
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if train_split_t and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t and train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
        if train is False and train_split_t is not None:
            plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
            
        if simulation:
            legend_list = ["{}-dim Simu".format(dim+1), "{}-dim MLP".format(dim+1)]
            plt.legend(legend_list)
            plt.title("Epoch - {} - Drift Term (Dim {}) Evaluated through time".format(epoch+1, dim+1))
        else:
            legend_list = ["{}-dim data".format(dim+1), "{}-dim MLP".format(dim+1)]
            if train_split_t:
                legend_list.append("{}-dim Forecast".format(dim+1))
                
            plt.legend(legend_list)
            plt.title("Epoch - {} - Mean Path (Dim {}) Evaluated through time".format(epoch+1, dim+1))
    if test == False:        
        plt.savefig(plot_savepath + "{}_Epoch_{}.pdf".format(("Train" if train else "Val"), epoch+1),
                    bbox_inches='tight', format="pdf")
    else:
        plt.savefig(plot_savepath + "{}_test_mean_drift.pdf".format(head),bbox_inches='tight', format="pdf")
    plt.close("all")
    
    
    
def plot_gradient(xs, 
                  driftMLP,
                  device,
                  ts,
                  data_params,
                  plot_savepath,
                  truth,
                  snap_time=100,
                  generative=False,
                  plot_scale=2, 
                  test=False,
                  head="None", 
                  train_particle_labels=None,
                  plot_kde=True,
                  samples=None,
                  ax = None):
    if ax is None:
        has_ax = False
        fig = plt.figure(figsize=(4,4), dpi=200)
        ax = plt.subplot(111)
    else:
        has_ax = True
    if plot_kde == True:
        if samples is not None:
            ax = sns.kdeplot(x=samples[:,0], y=samples[:,1], shade=True, cmap = "Reds", ax=ax)
        else:     
            ax = sns.kdeplot(x=xs[:,snap_time,0], y=xs[:,snap_time,1], shade=True, cmap = "Reds", ax=ax)
    
    x_center = (np.max(xs[:,snap_time, 0]) + np.min(xs[:, snap_time, 0]))/2
    r_x = np.max(xs[:,snap_time, 0]) - x_center
    x_low = x_center - r_x*plot_scale; x_high = x_center + r_x*plot_scale
    
    y_center = (np.max(xs[:,snap_time, 1]) + np.min(xs[:, snap_time, 1]))/2
    r_y = np.max(xs[:,snap_time, 1]) - y_center
    y_low = x_center - r_y*plot_scale; y_high = y_center + r_y*plot_scale
    
    x = np.linspace(x_low, x_high, 20)
    y = np.linspace(y_low, y_high, 20)
    
    X, Y = np.meshgrid(x, y)
    if xs.shape[-1] == 2:
        grids = np.concatenate((X.reshape(X.shape[0]*X.shape[1],1),Y.reshape(Y.shape[0]*Y.shape[1],1)),axis=-1)
        grids_reshape = grids.reshape(grids.shape[0],1,xs.shape[-1])
        grids_time = np.tile(grids_reshape, (1,100,1))
        if driftMLP.label_x == 1:
            grids_particle_label = make_particle_label(grids_time[:,0,:], partition=data_params["partition"])
            grids_data = torch.utils.data.TensorDataset(torch.from_numpy(grids_time), grids_particle_label)
        else:
            grids_data = torch.utils.data.TensorDataset(torch.from_numpy(grids_time))
            
        grids_loader = torch.utils.data.DataLoader(grids_data, batch_size=5, shuffle=False)
        
        start = 0
        drift = np.zeros(grids_time.shape)
        for val_idx, data in enumerate(grids_loader):
            end = start + data[0].shape[0]
            try:
                label_x = data[1].float().to(device) if driftMLP.label_x else None
            except IndexError:
                label_x = None
            if truth is False:
                drift[start:end] = MLP_drift(driftMLP, y_samps=torch.from_numpy(xs), x_obs=data[0], device=device,
                                            label_x = label_x, label_y = train_particle_labels,
                                            ts=torch.from_numpy(ts)).detach().cpu().numpy()
            elif truth is True and generative is False:
                drift[start:end] = simu_drift(**data_params, y_samps=torch.from_numpy(xs).detach().cpu(), 
                                            x_obs=data[0].detach().cpu(), 
                                            ts=torch.from_numpy(ts))
            start = end
            
        if generative is True and truth is True:
            drift[:,snap_time,:] = BB_get_drift(data_params["datatype"], grids=grids_time[:,snap_time,:],T=data_params["T"], d=grids_time.shape[-1])
        
        drift_strength = (drift[:,snap_time,0].reshape(X.shape[0], X.shape[1])*\
                    drift[:,snap_time,1].reshape(Y.shape[0], Y.shape[1]))
        drift_strength_minmax = ((drift_strength - drift_strength.min())/(drift_strength.max() - drift_strength.min()))
        
        drift_strength_scale = 2
        if 'Fitz' in head and truth == False:
            drift_strength_scale = 3
        elif 'Fitz' in head and truth == True:
            drift_strength_scale = 2
        elif 'kura' in head:
            drift_strength_scale = 2.3
        
        ax.streamplot(X, Y, 
                    drift[:,snap_time,0].reshape(X.shape[0], X.shape[1]),
                    drift[:,snap_time,1].reshape(Y.shape[0], Y.shape[1]), 
                    color="grey", density=2, arrowsize=0.5, arrowstyle="-|>",
                    linewidth=drift_strength_scale*drift_strength_minmax)

    bound = plot_scale-0.5
    
    x_low_bound = x_center - r_x*bound; x_high_bound = x_center + r_x*bound
    y_low_bound = x_center - r_y*bound; y_high_bound = y_center + r_y*bound
    
    ax.axis([int(x_low_bound), int(x_high_bound),
             int(y_low_bound), int(y_high_bound)])
            
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    ax.spines['left'].set_bounds(int(y_low_bound), int(y_high_bound))
    ax.spines['left'].set_position(('outward', 5))
    
    ax.spines['bottom'].set_bounds(int(x_low_bound), int(x_high_bound))
    ax.spines['bottom'].set_position(('outward', 5))
    
    ax.set_xticks([int(x_low_bound), int(x_high_bound)]) 
    ax.set_yticks([int(y_low_bound), int(y_high_bound)])
    if has_ax is False:
        if plot_kde == True:
            if test is False:
                plt.savefig(plot_savepath + "{}_Gradient_quiver.pdf".format(head),
                            bbox_inches='tight', 
                            format="pdf")
            elif truth is True:
                plt.savefig(plot_savepath + "True_{}_t={}_Gradient_quiver.pdf".format(head, snap_time),
                            bbox_inches='tight', 
                            format="pdf")
            else:
                plt.savefig(plot_savepath + "{}_t={}_Gradient_quiver.pdf".format(head, snap_time),bbox_inches='tight', format="pdf")
            plt.close()
        else:
            if test is False:
                plt.savefig(plot_savepath + "No_kde_{}_Gradient_quiver.pdf".format(head),
                            bbox_inches='tight', 
                            format="pdf")
            elif truth is True:
                plt.savefig(plot_savepath + "No_kde_True_{}_t={}_Gradient_quiver.pdf".format(head, snap_time),
                            bbox_inches='tight', 
                            format="pdf")
            else:
                plt.savefig(plot_savepath + "No_kde_{}_t={}_Gradient_quiver.pdf".format(head, snap_time),bbox_inches='tight', format="pdf")
            plt.close()
    else:
        return ax
    plt.close("all")
    
def plot_gradient_thru_t(xs, 
                         driftMLP,
                         device,
                         ts,
                         data_params,
                         plot_savepath,
                         truth,
                         plot_scale=2, 
                         test=False,
                         head="None"):
    t_list = list(range(0,xs.shape[1], int(xs.shape[1]/5))) + [xs.shape[1] - 1]
    fig, ax = plt.subplots(figsize=((len(t_list))*5, 5), nrows=1, ncols=len(t_list))
    
    for i, t in enumerate(t_list):
        ax[i] = plot_gradient(xs = xs, 
                              driftMLP = driftMLP,
                              device = device,
                              ts = ts,
                              data_params = data_params,
                              plot_savepath = plot_savepath,
                              truth = truth,
                              snap_time=t,
                              plot_scale=2, 
                              test=test,
                              head=head,
                              ax = ax[i])
        
    if test is False:
        plt.savefig(plot_savepath + "Thru_t_Gradient_quiver_{}.pdf".format(head),
                        bbox_inches='tight', 
                        format="pdf")
    elif truth is True:
        plt.savefig(plot_savepath + "True_Thru_t_{}_Gradient_quiver.pdf".format(head),
                        bbox_inches='tight', 
                        format="pdf")
    else:
        plt.savefig(plot_savepath + "Thru_t_{}_Gradient_quiver.pdf".format(head),bbox_inches='tight', format="pdf")
    plt.close("all")
    
    
def plot_ot_map(otmap,
                plot_particles,
                n_samples,
                plot_savepath,
                seed=None,
                truth=False,
                grid=False,
                average=True):
    fig, ax = plt.subplots(figsize=(5,3), ncols=1, nrows = 1, dpi=200)
    np.random.seed(seed)
    index = np.random.choice(list(range(n_samples)), plot_particles, False)
    if seed is None:
        index = list(range(n_samples))
    ax.scatter(otmap[index,0,0], otmap[index,0,1], color="blue")
    ax.scatter(otmap[index,-1,0], otmap[index,-1,1], color="red", marker="x")
    
    for i in index:
        ax.plot(otmap[i,:,0], otmap[i,:,1], color="grey")
    if truth == True and grid == False and average == False:
        plt.savefig(plot_savepath + "True_OT_map.pdf",bbox_inches='tight', format="pdf")
    elif truth == False and grid == False and average == True:
        plt.savefig(plot_savepath + "Average_Gen_OT_map.pdf",bbox_inches='tight', format="pdf")
    elif truth == False and grid == True and average == True:
        plt.savefig(plot_savepath + "Grid_Gen_OT_map.pdf",bbox_inches='tight', format="pdf")
    elif truth == False and grid == False and average == False:
        plt.savefig(plot_savepath + "Single_Gen_OT_map.pdf",bbox_inches='tight', format="pdf")
    
    plt.close("all")