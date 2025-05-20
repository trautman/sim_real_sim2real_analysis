import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import yaml
import os
import warnings

np.seterr(all='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)


import yaml
import numpy as np

def calling_data_files(file1, file2):
    # Correct the file list (was mistakenly using file24)
    files = [file1, file2]

    data_list = []
    for f in files:
        with open(f) as fp:
            data_list.append(yaml.safe_load(fp))

    # INITIALIZE LISTS for 8 keys (one per file)
    sites, states, names, metrics, means, stds, Ns, bins_ = [[] for _ in range(8)]

    array_keys = {'mean', 'std'}
    keys = ['site', 'state', 'baseline', 'metric', 'mean', 'std', 'N', 'bin']
    containers = [sites, states, names, metrics, means, stds, Ns, bins_]
    for data in data_list:
        for key, container in zip(keys, containers):
            value = np.array(data[key]) if key in array_keys else data[key]
            container.append(value)

    # Determine the minimum length among the "mean" arrays
    index = min(len(m) for m in means)

    # Copy the full arrays as "true" values before truncation
    means_true = [m.copy() for m in means]
    stds_true  = [s.copy() for s in stds]
    bins_true  = [np.array(b) for b in bins_]
    Ns_true    = [np.array(n) for n in Ns]

    # Truncate all arrays to the minimum index for consistency
    for i in range(len(means)):
        means[i] = means[i][:index]
        stds[i]  = stds[i][:index]
        bins_[i] = bins_[i][:index]
        Ns[i]    = Ns[i][:index]

    # Now, we assume there are only two conditions in these files.
    site_1, site_2 = sites
    state_1, state_2 = states
    name_1, name_2 = names
    metric_1, metric_2 = metrics

    m_1, m_2 = means
    std_1, std_2 = stds
    b_1, b_2 = bins_
    N_1, N_2 = Ns

    m_1_true, m_2_true = means_true
    std_1_true, std_2_true = stds_true
    b_1_true, b_2_true = bins_true
    N_1_true, N_2_true = Ns_true

    # Return the data in the same order as expected by your call (24 values)
    return (site_1, state_1, name_1, metric_1, m_1, std_1, b_1, N_1,
            site_2, state_2, name_2, metric_2, m_2, std_2, b_2, N_2,
            m_1_true, std_1_true, b_1_true, N_1_true, m_2_true, std_2_true, b_2_true, N_2_true)


def ttest_p_distribution(m_1, m_2, std_1, std_2, N_1, N_2, trials=1000):
    np.random.seed(42)  # Set seed for reproducibility

    p_distributions = [[] for _ in range(min(len(m_1), len(m_2)))]  # Create empty lists for each bin

    for _ in range(trials):  # Run multiple trials
        for bin in range(min(len(m_1), len(m_2))):
            # Generate random samples for both groups
            rvs1 = stats.norm.rvs(loc=m_1[bin], scale=std_1[bin], size=N_1[bin])
            rvs2 = stats.norm.rvs(loc=m_2[bin], scale=std_2[bin], size=N_2[bin])

            # Perform t-test
            t_stat, p = stats.ttest_ind(rvs1, rvs2)

            # Store p-value for this trial
            p_distributions[bin].append(p)

    return p_distributions



# COMPUTING HEDGES G BETWEEN SIM V REAL, CAN BE BRNE VS DWB;
# BASELINES CAN BE 2 ALGORITHMS OR CAN BE SIM V REAL
def hedges_g(m_1, m_2, std_1, std_2, N_1, N_2, compare_name):
  # inputs must match in domain since its a comparison
  s1, s2 = np.power(std_1, 2), np.power(std_2, 2)
  with np.errstate(divide='ignore', invalid='ignore'):
    s = np.sqrt(((N_1 - 1) * s1 + (N_2 - 1) * s2) / (N_1 + N_2 - 2))
  g = (m_1 - m_2) / s
  g = np.around(g, 2)
  J = 1 - (3 / (4 * (N_1 + N_2) - 9))
  g = np.around(J * g, 2)
  # print('BRNE compared to {}'.format(compare_name))
  # print('g corrected {}'.format(g))

  se_g = np.around(np.sqrt((N_1 + N_2) / (N_1 * N_2) + (g**2) / \
                            (2 * (N_1 + N_2))), 2)
  confidence = 0.95
  z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
  ci_upper = np.around(g + z_value * se_g, 3)
  ci_lower = np.around(g - z_value * se_g, 3)

  return g, se_g, z_value, ci_upper, ci_upper



def lin_regress(name_1, name_2, metric_1, metric_2, \
  bin_arr_1, bin_arr_2, mean_arr_1, mean_arr_2, g_s2r, \
  index_s2r, site_1, site_2):

  main_data = [(bin_arr_1, mean_arr_1), (bin_arr_2, mean_arr_2)]

  slope_lins, intercept_lins, r_lins, p_lins = [], [], [], []
  r_spears, p_spears = [], []

  for bins, means in main_data:
      res = stats.linregress(bins, means)
      slope_lins.append(round(res.slope, 2))
      intercept_lins.append(round(res.intercept, 3))
      p_lins.append(round(res.pvalue, 3))
      r_lins.append(round(res.rvalue, 2))

      r_spear, p_spear = spearmanr(bins, means)
      p_spears.append(p_spear)
      r_spears.append(r_spear)

  slope_1, slope_2 = slope_lins
  intercept_1, intercept_2 = intercept_lins
  p_lin_1, p_lin_2 = p_lins
  r_lin_1, r_lin_2 = r_lins
  p_spear_1, p_spear_2 = p_spears
  r_spear_1, r_spear_2 = r_spears


  res_g_s2r = stats.linregress(bin_arr_1[:index_s2r], g_s2r)
  slope_g_s2r = round(res_g_s2r.slope, 3)
  intercept_g_s2r = round(res_g_s2r.intercept, 3)
  p_lin_g_s2r = round(res_g_s2r.pvalue, 3)
  r_lin_g_s2r = round(res_g_s2r.rvalue, 2)
  r_spear_g_s2r, p_spear_g_s2r = spearmanr(bin_arr_1[:index_s2r], g_s2r)

  return slope_1, slope_2, slope_g_s2r, \
         intercept_1, intercept_2, intercept_g_s2r, \
         p_lin_1, p_lin_2, p_lin_g_s2r, \
         r_lin_1, r_lin_2, r_lin_g_s2r, \
         p_spear_1, p_spear_2, p_spear_g_s2r, \
         r_spear_1, r_spear_2, r_spear_g_s2r



# ===============================
# VISUALIZATION
# PLOTTING BASELINES ERROR AND LINE FIT

# Import these for the custom legend handling of error bars
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

def error_and_line_fit_baseline(
    name_1, name_2, \
    site_1, site_2, \
    state_1, state_2, \
    metric_1, metric_2, \
    b_1, b_2, \
    m_1, m_2, \
    std_1, std_2, \
    s_1, s_2, \
    i_1, i_2, \
    r_1, r_2, \
    p_1, p_2, \
    N_1, N_2, \
    r_spear_1, r_spear_2, \
    p_spear_1, p_spear_2, \
    ax, \
    fontsize, \
    error_marker_size, \
    regression_linewidth, \
    error_capsize, \
    legend_fontsize, \
    correlation_threshold, \
    performance_title, ci_linewidth, legend_location, xy_label_size, iv):


    # COMPUTING STE USING STD
    ste_1 = std_1 / np.sqrt(N_1)
    ste_2 = std_2 / np.sqrt(N_2)

    # Group data in lists for iteration
    names      = [name_1, name_2]
    metrics    = [metric_1, metric_2]
    states     = [state_1, state_2]
    b_values   = [b_1, b_2]
    m_values   = [m_1, m_2]
    ste_values = [ste_1, ste_2]
    colors     = ['blue', 'magenta', 'red', 'green']

    # Regression-related
    intercepts    = [i_1, i_2]
    slopes        = [s_1, s_2]
    pearson_r     = [r_1, r_2]
    pearson_p     = [p_1, p_2]
    spearman_r    = [r_spear_1, r_spear_2]
    spearman_p    = [p_spear_1, p_spear_2]

    # We'll store the ErrorbarContainer objects and regression lines here
    error_handles = []
    regression_handles = []

    # 1. Plot the error bars
    for i in range(2):
        label_text = f"{states[i]} performance $\mu \pm 95\%$CI"

        # Capture the entire ErrorbarContainer instead of just (line, cap, bar)
        error_container = ax.errorbar(
            b_values[i],
            m_values[i],
            yerr=1.96 * ste_values[i],
            fmt='o',
            markersize=error_marker_size,
            linewidth=ci_linewidth,
            capsize=error_capsize,
            color=colors[i],
            linestyle='',
            label=label_text  # label will go on the error bar container
        )
        error_handles.append(error_container)

        # 2. Check correlations and plot regression lines if needed
        if pearson_p[i] < 0.05:
            label = (f"{names[i]} nominal=${intercepts[i]:.1f}m$, "
                    f"$\\Delta_{{perf}}={slopes[i]:.1f}\\Delta m/\\Delta\\rho$,"
                    f" $r={pearson_r[i]:.1f}$")
            reg_line, = ax.plot(
                b_values[i],
                intercepts[i] + slopes[i] * b_values[i],
                '--',
                linewidth=regression_linewidth,
                label=label,
                color=colors[i]
            )
        elif spearman_p[i] < 0.05:
            label = f"{names[i]} monotonic correlation $r_s={spearman_r[i]:.2f}$"
            reg_line, = ax.plot(
                [], [], 'r--',
                linewidth=regression_linewidth,
                label=label,
                color=colors[i]
            )
        else:
            label = (f"{names[i]} no correlation")
            reg_line, = ax.plot(
                [], [], 'r--',
                linewidth=regression_linewidth,
                label=label,
                color=colors[i]
            )
        regression_handles.append(reg_line)
        # #############PRINT REGRESSION PARAMETERS
        print(f"PERF {metric_1}, {site_1}, {states[i]}, {names[i]} "
              f"p_l = {pearson_p[i]:.1f}, "
              f"beta_0={intercepts[i]:.1f}, "
              f"d_perf/d_rho= {slopes[i]:.1f}, "
              f"r_l={pearson_r[i]:.1f}, "
              f"p_m = {spearman_p[i]:.1f}"
              f"r_m={spearman_r[i]:.1f}, ")



    #  ERROR AND REGRESSION HANDLES
    # handles = error_handles + regression_handles
    # labels = [h.get_label() for h in error_handles] + \
    #           [rh.get_label() for rh in regression_handles]
    # REGRESSION HANDLES ONLY
    # handles = regression_handles
    # labels = [rh.get_label() for rh in regression_handles]
    # ERROR HANDLES ONLY
    handles = error_handles
    labels = [h.get_label() for h in error_handles]

    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=legend_fontsize,
        handler_map={ErrorbarContainer: HandlerErrorbar()},
        loc=legend_location
    )

    # 5. Set plot title and axis labels
    ax.set_title(f"{performance_title}, {site_1}, {name_1}", fontsize=fontsize)
    if "max" in iv:
        ax.set_xlabel("Max Crowd Density (people$/m^2$)", fontsize=fontsize)
    else:
        ax.set_xlabel("Mean Crowd Density (people$/m^2$)", fontsize=fontsize)
    ax.set_ylabel(metric_1, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=xy_label_size)




# PLOTTING G ERROR AND LINE FIT, AXES, TITLE
# def error_and_line_fit_g(
#     name_1, name_2, name_3, name_4,
#     site_1, site_2, site_3, site_4,
#     metric_1, metric_2, metric_3, metric_4,
#     b_1, b_2, b_3, b_4,
#     g_brne_dwb, g_brne_human, g_brne_teleop,
#     se_g_brne_dwb, se_g_brne_human, se_g_brne_teleop,
#     s_g_brne_dwb, s_g_brne_human, s_g_brne_teleop,
#     i_g_brne_dwb, i_g_brne_human, i_g_brne_teleop,
#     r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop,
#     p_lin_g_brne_dwb, p_lin_g_brne_human, p_lin_g_brne_teleop,
#     ax, index_bd, index_bh, index_bt,
#     r_g_spear_brne_dwb, p_g_spear_brne_dwb,
#     r_g_spear_brne_human, p_g_spear_brne_human,
#     r_g_spear_brne_teleop, p_g_spear_brne_teleop,
#     fontsize,
#     error_marker_size,
#     regression_linewidth,
#     error_capsize,
#     legend_fontsize,
#     lim_x,
#     if_g_line_fit,
#     correlation_threshold,
#     performance_gap_title,
#     ci_threshold, ci_linewidth, legend_location, xy_label_size, iv
# ):
def error_and_line_fit_g(
                        name_1, name_2, 
                        site_1, site_2,
                        metric_1, metric_2,
                        b_1, b_2,
                        index_s2r,
                        g_s2r, 
                        se_s2r,
                        s_g_s2r, 
                        i_g_s2r,
                        r_lin_g_s2r,
                        p_lin_g_s2r, 
                        r_spear_g_s2r, 
                        p_spear_g_s2r,
                        ax,
                        fontsize,
                        error_marker_size,
                        regression_linewidth,
                        error_capsize,
                        legend_fontsize,
                        lim_x,
                        if_g_line_fit,
                        correlation_threshold,
                        performance_gap_title,
                        ci_threshold, 
                        ci_linewidth, 
                        legend_location, 
                        xy_label_size, 
                        iv):
      
    def filter_ci(x_vals, y_vals, se_vals, idx):
        x_slice = x_vals[:idx]
        y_slice = y_vals[:idx]
        se_slice = se_vals[:idx]

        if ci_threshold is not None:
            ci_95 = 1.96 * se_slice
            mask = ci_95 <= ci_threshold
            return x_slice[mask], y_slice[mask], se_slice[mask]
        else:
            return x_slice, y_slice, se_slice

    b_1_s2r, g_s2r, se_s2r = filter_ci(b_1, g_s2r, se_s2r, index_s2r)

    error_container1 = ax.errorbar(
        b_1_s2r, g_s2r, yerr=1.96 * se_s2r,
        fmt='o', markersize=error_marker_size,
        linewidth=ci_linewidth,
        capsize=error_capsize,
        linestyle='-',
        # label="Sim2Real $g$ Mean ± 95% CI",
        color='magenta'
    )
   
    error_handles = [error_container1]
    regression_handles = []
    threshold_handles = []

    if if_g_line_fit == 'True':
        if p_lin_g_s2r < 0.05:
            label_s2r = (f"S2R nominal$={round(i_g_s2r,2)}$, "
                         f"$\\Delta_{{gap}}={round(s_g_s2r, 2)}"
                         f"\\Delta g/\\Delta \\rho$, "
                         f"$r={r_lin_g_s2r}$")
            lines2r, = plt.plot(b_1_s2r, i_g_s2r + s_g_s2r * b_1_s2r,
                'k--',
                linewidth=regression_linewidth,
                label=label_s2r,
                color='magenta'
            )
        elif p_spear_g_s2r < 0.05:
            lines2r, = plt.plot([], [], 'k--',
                label=(f"$g$ S2R monotonic correlation "
                      f"$r_s={round(r_spear_g_s2r, 2)}") ,
                linewidth=regression_linewidth,
                color='magenta'
            )
        else:
            lines2r, = plt.plot([], [], 'k--',
                label="$g$ S2R no correlation",
                linewidth=regression_linewidth,
                color='magenta'
            )

        regression_handles.append(lines2r)
        # PRINT REGRESSION PARAMETERS S2R
        print(f"GAP SIM2REAL {metric_1} {site_1} "
            f"p_l^g={p_lin_g_s2r:.1f} "
            f"beta_0^g={i_g_s2r:.1f}, "
            f"d_gap/d_rho={s_g_s2r:.1f}, "
            f"r_l^g={r_lin_g_s2r:.1f}, "
            f"p_m^g={p_spear_g_s2r:.1f}, "
            f"r_m^g={r_spear_g_s2r:.1f}, ")



        # --- Threshold lines ---
        thresholdupper, = ax.plot(
            [0, lim_x], [0.5, 0.5], color='k', linestyle='-', lw=3.0,
            label='$g>0.5$ sim2real gap not closed'
        )
        thresholdlower, = ax.plot(
            [0, lim_x], [-0.5, -0.5], color='k', linestyle='-', lw=3.0,
            label='$g<-0.5$ sim2real gap not closed'
        )
        threshold_handles.extend([thresholdupper, thresholdlower])

        
        handles = error_handles + threshold_handles
        bin_label = f'Sim2real gap $\mu \pm 95\%$CI' 
        labels = [bin_label] \
            + [thresholdupper.get_label(), thresholdlower.get_label()]

    else: # IF NO REGRESSION IS CONDUCTED
        thresholdupper, = ax.plot(
                                [0, lim_x], [0.5, 0.5],
                                color='k',
                                linestyle='-',
                                linewidth=regression_linewidth,
                                label='$g>0.5$ sim2real gap not closed')
        thresholdlower, = ax.plot(
                                [0, lim_x], [-0.5, -0.5],
                                color='k',
                                linestyle='-',
                                linewidth=regression_linewidth,
                                label='$g<-0.5$ sim2real gap not closed')
        threshold_handles.extend([thresholdupper, thresholdlower])

        bin_label = 'Sim2Real Gap $\mu \pm 95\%$CI'
        handles = error_handles + threshold_handles
        labels = [
                bin_label,
                thresholdupper.get_label(),
                thresholdlower.get_label()]

    ax.legend(
            handles=handles,
            labels=labels,
            fontsize=legend_fontsize,
            handler_map={ErrorbarContainer: HandlerErrorbar()},
            loc=legend_location)

    ax.set_title(f"{performance_gap_title}, {site_1}", fontsize=fontsize)
    if "max" in iv:
        ax.set_xlabel("Max Crowd Density (people$/m^2$)", fontsize=fontsize)
    else:
        ax.set_xlabel("Mean Crowd Density (people$/m^2$)", fontsize=fontsize)
    ax.set_ylabel(f"{metric_1} Sim2Real Gap", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=xy_label_size)






















