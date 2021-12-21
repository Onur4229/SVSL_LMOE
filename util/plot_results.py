import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from matplotlib.gridspec import GridSpec

colors = ['blue', 'red', 'green', 'purple' ,'black', 'gray', 'rosybrown', 'maroon', 'red', 'coral', 'chocolate',
          'darkorange',  'olive', 'lightgreen', 'limegreen', 'lightseagreen', 'cyan', 'lightblue', 'deepskyblue', 'lightslategray',
           'blueviolet', 'plum', 'magenta', 'palevioletred']*50

def plot_cmp_ctxts_4d(model, fig, colors, cmp_indices):
    if fig is None:
        fig = plt.figure()
        fig_axes_1 = fig.add_subplot(211)
        fig_axes_2 = fig.add_subplot(212)
    if cmp_indices is None:
        cmp_indices = range(model.num_components)
    for j, i in enumerate(cmp_indices):
        c_comp = model.ctxt_components[i]
        color = colors[i] if colors is not None else None
        c_comp_1_mean = c_comp.mean[:2]
        c_comp_1_cov = c_comp.covar[:2, :2]
        draw_2d_gaussian(c_comp_1_mean, c_comp_1_cov, fig_axes_1, color=color)
        c_comp_2_mean = c_comp.mean[2:]
        c_comp_2_cov = c_comp.covar[2:, 2:]
        draw_2d_gaussian(c_comp_2_mean, c_comp_2_cov, fig_axes_2, color=color)
    return fig

def plot_cmp_ctxts(model, fig=None, colors=None, cmp_indices=None, ctxt_range_min=-6, ctxt_range_max=6, swap_axis=False):
    ctxt_dim = model.ctxt_dim
    if ctxt_dim == 4:
        plot_cmp_ctxts_4d(model, fig, colors, cmp_indices)
    else:
        if fig is None:
            fig = plt.figure()
            plt.grid(True)
        plt.figure(fig.number)
        if cmp_indices is None:
            cmp_indices = range(model.num_components)

        for j, i in enumerate(cmp_indices):
            c_comp = model.ctxt_components[i]
            color = colors[i] if colors is not None else None
            if ctxt_dim == 2:
                draw_2d_gaussian(c_comp.mean, c_comp.covar, fig, color=color)
            elif ctxt_dim == 1:
                draw_1d_gaussian(c_comp.mean, sigma=c_comp.covar, range_min=ctxt_range_min, range_max=ctxt_range_max, fig=fig,
                                 swap_axis=swap_axis, color=color)
            else:
                raise ValueError("Cannot plot more than 2 dimensions")
    return fig

def draw_2d_gaussian(mu, sigma, fig, plt_std = 2, *args, **kwargs):
    try:
        plt.figure(fig.number)
    except Exception:
        plt.sca(fig)
    idx = np.where(np.abs(sigma)<1e-10)
    if idx[0].shape[0] and idx[1].shape[0]:
        sigma[idx] = 0
    (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(sigma)
    phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])
    plt.plot(mu[0:1], mu[1:2], marker="x", *args, **kwargs)

    a = plt_std * np.sqrt(largest_eigval)
    b = plt_std * np.sqrt(smallest_eigval)

    ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi, num=20))
    ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi, num=20))

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
    plt.plot(mu[0] + r_ellipse[:, 0], mu[1] + r_ellipse[:, 1], *args, **kwargs)

def draw_context_bounds_2d(context_range_bounds, fig):
    try:
        plt.figure(fig.number)
    except Exception:
        plt.sca(fig)
    plt.plot([context_range_bounds[0][0], context_range_bounds[1][0]],
             [context_range_bounds[0][1], context_range_bounds[0][1]], 'r-', linewidth=1)
    plt.plot([context_range_bounds[0][0], context_range_bounds[1][0]],
             [context_range_bounds[1][1], context_range_bounds[1][1]], 'r-', linewidth=1)
    plt.plot([context_range_bounds[0][0], context_range_bounds[0][0]],
             [context_range_bounds[0][1], context_range_bounds[1][1]], 'r-', linewidth=1)
    plt.plot([context_range_bounds[1][0], context_range_bounds[1][0]],
             [context_range_bounds[0][1], context_range_bounds[1][1]], 'r-', linewidth=1)

def draw_1d_gaussian(mu, sigma, range_min, range_max, fig, swap_axis=False, n_points = 1000, grid=True, *args, **kwargs):
    points = np.linspace(range_min, range_max, n_points)
    densities = multivariate_normal.pdf(points, mean=mu, cov=sigma)
    plt.figure(fig.number)
    if swap_axis:
        x_axis = densities - 5
        y_axis = points
    else:
        x_axis = points
        y_axis = densities
    plt.plot(x_axis, y_axis, *args, **kwargs)
    if grid:
        plt.grid()

def plot_ball_trajs(ball_trajs_list, fig=None, idx=None, use_colors=None):
    if fig is None:
        fig = plt.figure()
        gs = GridSpec(3, 9, figure=fig)
        fig.add_subplot(gs[0, :-1])
        plt.grid()
        fig.add_subplot(gs[1, :-1])
        plt.grid()
        fig.add_subplot(gs[2, :-1])
        plt.grid()
        fig.add_subplot(gs[:, -1])
    if use_colors is None:
        use_colors = colors

    plt.figure(fig.number)
    c_axes = fig.axes
    n_elems = 0
    used_labels = []
    for k, trajs in enumerate(ball_trajs_list):
        if type(trajs) is list:
            n_elems = len(trajs)
            for c_cmp_traj in trajs:
                c_axes[0].plot(c_cmp_traj[:, 0], color=use_colors[k], alpha=0.5, label=str(k))
                c_axes[1].plot(c_cmp_traj[:, 1], color=use_colors[k], alpha=0.5, label=str(k))
                c_axes[2].plot(c_cmp_traj[:, 2], color=use_colors[k], alpha=0.5, label=str(k))
        else:
            if idx is not None:
                c_label = str(idx[k])
                c_color = use_colors[int(idx[k])]
            else:
                c_label = str(k)
                c_color = use_colors[k]
            if c_label in used_labels:
                c_label = None
            else:
                used_labels.append(c_label)
            n_elems = 1
            c_axes[0].plot(trajs[:, 0], color=c_color, alpha=0.5, label=c_label)
            c_axes[1].plot(trajs[:, 1], color=c_color, alpha=0.5, label=c_label)
            c_axes[2].plot(trajs[:, 2], color=c_color, alpha=0.5, label=c_label)
    handles, labels = c_axes[2].get_legend_handles_labels()
    c_axes[-1].legend(handles[::n_elems], labels[::n_elems])
    return fig