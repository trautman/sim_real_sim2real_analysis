import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import yaml
import os
import warnings

np.seterr(all='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)


def calling_data_files(file1, file2, file3, file4):
    files = [file1, file2, file3, file4]

    data_list = []
    for f in files:
        with open(f) as fp:
            data_list.append(yaml.safe_load(fp))

    # INITIALIZE LISTS
    sites, states, names, metrics, means, stds, Ns, bins_ = \
                                                        [[] for _ in range(8)]

    array_keys = {'mean', 'std'}
    keys = ['site', 'state', 'baseline', 'metric', 'mean', 'std', 'N', 'bin']
    containers = [sites, states, names, metrics, means, stds, Ns, bins_]
    for data in data_list:
        for key, container in zip(keys, containers):
            value = np.array(data[key]) if key in array_keys else data[key]
            container.append(value)

    index = min(len(m) for m in means)

    means_true = [m.copy() for m in means]
    stds_true = [s.copy() for s in stds]
    bins_true = [np.array(b) for b in bins_]
    Ns_true   = [np.array(n) for n in Ns]

    for i in range(len(means)):
        means[i] = means[i][:index]
        stds[i]  = stds[i][:index]
        bins_[i] = bins_[i][:index]
        Ns[i]    = Ns[i][:index]

    site_1, site_2, site_3, site_4 = sites
    state_1, state_2, state_3, state_4 = states
    name_1, name_2, name_3, name_4 = names
    metric_1, metric_2, metric_3, metric_4 = metrics

    m_1, m_2, m_3, m_4 = means
    std_1, std_2, std_3, std_4 = stds
    b_1, b_2, b_3, b_4 = bins_
    N_1, N_2, N_3, N_4 = Ns

    m_1_true, m_2_true, m_3_true, m_4_true = means_true
    std_1_true, std_2_true, std_3_true, std_4_true = stds_true
    b_1_true, b_2_true, b_3_true, b_4_true = bins_true
    N_1_true, N_2_true, N_3_true, N_4_true = Ns_true

    # 7. Return the exact same structure as your original function
    return (site_1, state_1, name_1, metric_1,
      m_1, std_1, b_1, N_1, site_2, state_2, name_2, metric_2, m_2, std_2, b_2, \
      N_2, site_3, state_3, name_3, metric_3, m_3, std_3, b_3, N_3, site_4, \
      state_4, name_4, metric_4, m_4, std_4, b_4, N_4, m_1_true, std_1_true, \
      b_1_true, N_1_true, m_2_true, std_2_true, b_2_true, N_2_true, m_3_true, \
      std_3_true, b_3_true, N_3_true, m_4_true, std_4_true, b_4_true, N_4_true)



# COMPUTING HEDGES G BETWEEN SIM V REAL, CAN BE BRNE VS DWB;
# BASELINES CAN BE 2 ALGORITHMS OR CAN BE SIM V REAL
# def hedges_g(m_1, m_2, std_1, std_2, N_1, N_2, compare_name):
#   # inputs must match in domain since its a comparison
#   s1, s2 = np.power(std_1, 2), np.power(std_2, 2)
#   with np.errstate(divide='ignore', invalid='ignore'):
#     s = np.sqrt(((N_1 - 1) * s1 + (N_2 - 1) * s2) / (N_1 + N_2 - 2))
#   g = (m_1 - m_2) / s
#   g = np.around(g, 2)
#   J = 1 - (3 / (4 * (N_1 + N_2) - 9))
#   g = np.around(J * g, 2)
#   # print('BRNE compared to {}'.format(compare_name))
#   # print('g corrected {}'.format(g))

#   se_g = np.around(np.sqrt((N_1 + N_2) / (N_1 * N_2) + (g**2) / \
#                             (2 * (N_1 + N_2))), 2)
#   confidence = 0.95
#   z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
#   ci_upper = np.around(g + z_value * se_g, 3)
#   ci_lower = np.around(g - z_value * se_g, 3)

#   return g, se_g, z_value, ci_upper, ci_lower

def hedges_g(m_1, m_2, std_1, std_2, N_1, N_2, compare_name):
    # 1) pooled variance & SD
    s1, s2 = std_1**2, std_2**2
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.sqrt(((N_1 - 1)*s1 + (N_2 - 1)*s2) / (N_1 + N_2 - 2))

    # 2) Cohen's d (no rounding yet)
    # d = (m_1 - m_2) / s
    d = (m_1 - m_2)/std_1
    # d = (m_1 - m_2)
    print('std 1 inside', std_1)
    # 3) J‐correction → Hedges' g (still full precision)
    J = 1 - (3 / (4*(N_1 + N_2) - 9))
    # g = J * d
    g = d

    # standard error of Delta
    se_delta = np.sqrt(
        std_2**2/(N_2 * std_1**2)
        + 1.0/N_1
    )

    # 95% CI delta
    confidence = 0.95
    z = stats.norm.ppf(1 - (1 - confidence)/2)
    ci_lower = g - z * se_delta
    ci_upper = g + z * se_delta
    
    # # 4) standard error OF G
    # se_g = np.sqrt((N_1 + N_2)/(N_1*N_2) + g**2/(2*(N_1 + N_2)))

    # # 5) 95% CI via normal approx
    # confidence = 0.95
    # z_value = stats.norm.ppf(1 - (1 - confidence)/2)
    # ci_lower = g - z_value*se_g
    # ci_upper = g + z_value*se_g

    # 6) round only final outputs G
    # g      = np.around(g,     2)
    # se_g   = np.around(se_g,   2)
    # ci_lower = np.around(ci_lower, 3)
    # ci_upper = np.around(ci_upper, 3)
    
    # DELTA 
    g      = np.around(g,     2)
    se_delta   = np.around(se_delta,   2)
    ci_lower = np.around(ci_lower, 3)
    ci_upper = np.around(ci_upper, 3)

    # ⚠️ here we now return (ci_lower, ci_upper) instead of (ci_upper, ci_upper)
    return g, se_delta, z, ci_lower, ci_upper

def lin_regress(name_1, name_2, name_3, name_4, metric_1, metric_2, \
  metric_3, metric_4, bin_arr_1, bin_arr_2, bin_arr_3, bin_arr_4, mean_arr_1, \
  mean_arr_2, mean_arr_3, mean_arr_4, g_brne_dwb, g_brne_human, g_brne_teleop,
  index_bd, index_bh, index_bt, site_1, site_2, site_3, site_4):

  main_data = [(bin_arr_1, mean_arr_1), (bin_arr_2, mean_arr_2),
               (bin_arr_3, mean_arr_3), (bin_arr_4, mean_arr_4)]

  slopes, intercepts, r_vals, p_linears = [], [], [], []
  r_spears, p_spears = [], []

  for bins, means in main_data:
      res = stats.linregress(bins, means)
      slopes.append(round(res.slope, 2))
      intercepts.append(round(res.intercept, 3))
      r_vals.append(round(res.rvalue, 2))
      p_linears.append(round(res.pvalue, 3))

      r_spear, p_spear = spearmanr(bins, means)
      r_spears.append(r_spear)
      p_spears.append(p_spear)

  slope_1, slope_2, slope_3, slope_4 = slopes
  intercept_1, intercept_2, intercept_3, intercept_4 = intercepts
  r_1, r_2, r_3, r_4 = r_vals
  p_linear_1, p_linear_2, p_linear_3, p_linear_4 = p_linears
  r_spear_1, r_spear_2, r_spear_3, r_spear_4 = r_spears
  p_spear_1, p_spear_2, p_spear_3, p_spear_4 = p_spears

  g_data = [(bin_arr_1[:index_bd], g_brne_dwb),
    (bin_arr_1[:index_bh], g_brne_human), (bin_arr_1[:index_bt], g_brne_teleop)]

  slopes_g, intercepts_g, r_vals_g, p_linears_g = [], [], [], []
  r_spears_g, p_spears_g = [], []

  for bins_g, means_g in g_data:
      res_g = stats.linregress(bins_g, means_g)
      slopes_g.append(round(res_g.slope, 3))
      intercepts_g.append(round(res_g.intercept, 3))
      r_vals_g.append(round(res_g.rvalue, 2))
      p_linears_g.append(round(res_g.pvalue, 3))

      r_g_spear, p_g_spear = spearmanr(bins_g, means_g)
      r_spears_g.append(r_g_spear)
      p_spears_g.append(p_g_spear)

  slope_g_brne_dwb, slope_g_brne_human, slope_g_brne_teleop = slopes_g
  intercept_g_brne_dwb, intercept_g_brne_human, intercept_g_brne_teleop = \
                                                                  intercepts_g
  r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop = r_vals_g
  p_linear_g_brne_dwb, p_linear_g_brne_human, p_linear_g_brne_teleop = \
                                                                    p_linears_g
  r_g_spear_brne_dwb, r_g_spear_brne_human, r_g_spear_brne_teleop = r_spears_g
  p_g_spear_brne_dwb, p_g_spear_brne_human, p_g_spear_brne_teleop = p_spears_g

  return slope_1, slope_2, slope_3, slope_4, slope_g_brne_dwb, \
    slope_g_brne_human, slope_g_brne_teleop, intercept_1, intercept_2, \
    intercept_3, intercept_4, intercept_g_brne_dwb, intercept_g_brne_human, \
    intercept_g_brne_teleop, r_1, r_2, r_3, r_4, r_g_brne_dwb, r_g_brne_human,\
    r_g_brne_teleop, p_linear_1, p_linear_2, p_linear_3, p_linear_4, \
    p_linear_g_brne_dwb, p_linear_g_brne_human, p_linear_g_brne_teleop, \
    r_spear_1, r_spear_2, r_spear_3, r_spear_4, p_spear_1, p_spear_2, \
    p_spear_3, p_spear_4, r_g_spear_brne_dwb, p_g_spear_brne_dwb, \
    r_g_spear_brne_human, p_g_spear_brne_human, r_g_spear_brne_teleop, \
    p_g_spear_brne_teleop



# COMPUTING P VALUE BETWEEN REAL M1,STD1,N1 AND M2,STD2,N2
# def ttest(m_1, m_2, std_1, std_2, N_1, N_2):
#   p_ttest = []
#   # np.random.seed(42)#unless seed is set you'll get different
#   # p values for different trials
#   for bin in range(min(len(m_1), len(m_2))):
#       rvs1 = stats.norm.rvs(loc=m_1[bin], scale=std_1[bin], \
#                             size=N_1[bin])
#       rvs2 = stats.norm.rvs(loc=m_2[bin], scale=std_2[bin], \
#                             size=N_2[bin])
#       t_stat, p = stats.ttest_ind(rvs1, rvs2)
#       p_ttest.append(round(p, 3))
#   return p_ttest
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




# ===============================
# VISUALIZATION
# PLOTTING BASELINES ERROR AND LINE FIT

# Import these for the custom legend handling of error bars
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar



def error_and_line_fit_baseline(
    name_1, name_2, name_3, name_4,
    site_1, site_2, site_3, site_4,
    state_1, state_2, state_3, state_4,
    metric_1, metric_2, metric_3, metric_4,
    b_1, b_2, b_3, b_4,
    m_1, m_2, m_3, m_4,
    std_1, std_2, std_3, std_4,
    s_1, s_2, s_3, s_4,
    i_1, i_2, i_3, i_4,
    r_1, r_2, r_3, r_4,
    p_1, p_2, p_3, p_4,
    N_1, N_2, N_3, N_4,
    ax,
    r_spear_1, r_spear_2, r_spear_3, r_spear_4,
    p_spear_1, p_spear_2, p_spear_3, p_spear_4,
    fontsize,
    error_marker_size,
    regression_linewidth,
    error_capsize,
    legend_fontsize,
    correlation_threshold,
    performance_title, ci_linewidth, legend_location, xy_label_size, iv, \
    bin_clip, cutoff_left, cutoff_right
):


    # COMPUTING STE USING STD
    ste_1 = std_1 / np.sqrt(N_1)
    ste_2 = std_2 / np.sqrt(N_2)
    ste_3 = std_3 / np.sqrt(N_3)
    ste_4 = std_4 / np.sqrt(N_4)

    # Group data in lists for iteration
    names      = [name_1, name_2, name_3, name_4]
    states     = [state_1, state_2, state_3, state_4]
    metrics    = [metric_1, metric_2, metric_3, metric_4]
    b_values   = [b_1, b_2, b_3, b_4]
    m_values   = [m_1, m_2, m_3, m_4]
    ste_values = [ste_1, ste_2, ste_3, ste_4]
    colors     = ['black', 'magenta', 'red', 'green']

    # Regression-related
    intercepts    = [i_1, i_2, i_3, i_4]
    slopes        = [s_1, s_2, s_3, s_4]
    pearson_r     = [r_1, r_2, r_3, r_4]
    pearson_p     = [p_1, p_2, p_3, p_4]
    spearman_r    = [r_spear_1, r_spear_2, r_spear_3, r_spear_4]
    spearman_p    = [p_spear_1, p_spear_2, p_spear_3, p_spear_4]

    # We'll store the ErrorbarContainer objects and regression lines here
    error_handles = []
    regression_handles = []

    # 1. Plot the error bars
    for i in range(4):
        if i == 0:
            label_text = f"{name_1}, human pedestrians"
        else:
            label_text = f"{name_1} in {states[i]} simulator"

        b = np.asarray(b_values[i])
        m = np.asarray(m_values[i])
        ste = ste_values[i]               # this should already be a numpy array
        mask = (b <= bin_clip)

        # now overwrite with clipped data
        b = b[mask]
        m = m[mask]
        ste = ste[mask]

        # Capture the entire ErrorbarContainer instead of just (line, cap, bar)
        # error_container = ax.errorbar(
        #     b_values[i],
        #     m_values[i],
        #     yerr=1.96 * ste_values[i],
        #     fmt='o',
        #     markersize=error_marker_size,
        #     linewidth=ci_linewidth,
        #     capsize=error_capsize,
        #     color=colors[i],
        #     linestyle='',
        #     label=label_text  # label will go on the error bar container
        # )

        # if states[i] == 'ORCA':
        #     stq = 50.0*ste
        #     print('ORCAA', stq)
        # else:
        #     stq = ste
        error_container = ax.errorbar(
            b, m,
            yerr=1.96 * ste,
            fmt='o',
            markersize=error_marker_size,
            linewidth=ci_linewidth,
            capsize=error_capsize,
            color=colors[i],
            clip_on=True,          # make sure the artist itself is clipped
            label=label_text
        )

        error_handles.append(error_container)

        # 2. Check correlations and plot regression lines if needed
        # if pearson_p[i] < 0.05:
        #     label = (f"{names[i]} nominal=${intercepts[i]:.1f}m$, "
        #             f"$\\Delta_{{perf}}={slopes[i]:.1f}\\Delta m/\\Delta\\rho$,"
        #             f" $r={pearson_r[i]:.1f}$")
        #     reg_line, = ax.plot(
        #         b_values[i],
        #         intercepts[i] + slopes[i] * b_values[i],
        #         '--',
        #         linewidth=regression_linewidth,
        #         label=label,
        #         color=colors[i]
        #     )

        if pearson_p[i] < 0.05:
            x_line = np.linspace(0, bin_clip, 100)
            reg_line, = ax.plot(
                x_line,
                intercepts[i] + slopes[i] * x_line,
                '--',
                linewidth=regression_linewidth,
                color=colors[i],
                clip_on=True
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
        print(f"PERF {states[i]} "
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
    ax.set_title(f"{performance_title} of {name_1} at {site_1}", fontsize=fontsize)
    if "max" in iv:
        ax.set_xlabel("Max Crowd Density (people$/m^2$)", fontsize=fontsize)
    else:
        ax.set_xlabel("Mean Crowd Density (people$/m^2$)", fontsize=fontsize)
    ax.set_ylabel(metric_1, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=xy_label_size)
    ax.set_xlim(cutoff_left, cutoff_right)




# PLOTTING G ERROR AND LINE FIT, AXES, TITLE
def error_and_line_fit_g(
    name_1, name_2, name_3, name_4,
    site_1, site_2, site_3, site_4,
    state_1, state_2, state_3, state_4,
    metric_1, metric_2, metric_3, metric_4,
    b_1, b_2, b_3, b_4,
    g_brne_dwb, g_brne_human, g_brne_teleop,
    se_g_brne_dwb, se_g_brne_human, se_g_brne_teleop,
    s_g_brne_dwb, s_g_brne_human, s_g_brne_teleop,
    i_g_brne_dwb, i_g_brne_human, i_g_brne_teleop,
    r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop,
    p_lin_g_brne_dwb, p_lin_g_brne_human, p_lin_g_brne_teleop,
    ax, index_bd, index_bh, index_bt,
    r_g_spear_brne_dwb, p_g_spear_brne_dwb,
    r_g_spear_brne_human, p_g_spear_brne_human,
    r_g_spear_brne_teleop, p_g_spear_brne_teleop,
    fontsize,
    error_marker_size,
    regression_linewidth,
    error_capsize,
    legend_fontsize,
    lim_x,
    if_g_line_fit,
    correlation_threshold,
    performance_gap_title,
    ci_threshold, ci_linewidth, legend_location, xy_label_size, iv, 
    bin_clip, cutoff_left, cutoff_right
):
    """
    Plots g-values for BRNE vs DWB, Human, Teleop with optional line fits,
    filtering out large CI points if ci_threshold is set.
    Ensures the legend shows both the marker and error-bar caps.
    """

    # Helper function for filtering data by CI threshold
    def filter_ci(x_vals, y_vals, se_vals, idx):
        """
        x_vals: e.g., b_1
        y_vals: e.g., g_brne_dwb
        se_vals: e.g., se_g_brne_dwb
        idx: integer slice limit (e.g., index_bd)

        Returns (x_filtered, y_filtered, se_filtered).
        """
        x_slice = x_vals[:idx]
        y_slice = y_vals[:idx]
        se_slice = se_vals[:idx]

        # If ci_threshold is given, filter out points with 95% CI > ci_threshold
        if ci_threshold is not None:
            ci_95 = 1.96 * se_slice
            mask = ci_95 <= ci_threshold
            return x_slice[mask], y_slice[mask], se_slice[mask]
        else:
            return x_slice, y_slice, se_slice

    # 1) Filter / slice each dataset
    b_1_bd, g_dwb, se_dwb = filter_ci(b_1, g_brne_dwb, se_g_brne_dwb, index_bd)
    b_3_bh, g_human, se_human = filter_ci(b_3, g_brne_human, \
                                                    se_g_brne_human, index_bh)
    b_4_bt, g_teleop, se_teleop = filter_ci(b_4, g_brne_teleop, \
                                                  se_g_brne_teleop, index_bt)

    # 2) Plot error bars using ErrorbarContainer
    
    mask_bd    = (b_1_bd   >= cutoff_left) & (b_1_bd   <= cutoff_right)
    mask_human = (b_3_bh   >= cutoff_left) & (b_3_bh   <= cutoff_right)
    mask_tel   = (b_4_bt   >= cutoff_left) & (b_4_bt   <= cutoff_right)

    b1 = np.array(b_1_bd)[mask_bd]
    g1 = np.array(g_dwb)[mask_bd]
    se1 = np.array(se_dwb)[mask_bd]

    b2 = np.array(b_3_bh)[mask_human]
    g2 = np.array(g_human)[mask_human]
    se2 = np.array(se_human)[mask_human]

    b3 = np.array(b_4_bt)[mask_tel]
    g3 = np.array(g_teleop)[mask_tel]
    se3 = np.array(se_teleop)[mask_tel]


    error_container1 = ax.errorbar(
        b1, g1, yerr=1.96*se1,
        fmt='o', markersize=error_marker_size,
        linewidth=ci_linewidth, capsize=error_capsize,
        linestyle='-',
        clip_on=True, color='magenta'
    )
    error_container2 = ax.errorbar(
        b2, g2, yerr=1.96*se2,
        fmt='o', markersize=error_marker_size,
        linewidth=ci_linewidth, capsize=error_capsize,
        linestyle='-',
        clip_on=True, color='red'
    )
    error_container3 = ax.errorbar(
        b3, g3, yerr=1.96*se3,
        fmt='o', markersize=error_marker_size,
        linewidth=ci_linewidth, capsize=error_capsize,
        linestyle='-',
        clip_on=True, color='green'
    )


    # error_container1 = ax.errorbar(
    #     b_1_bd, g_dwb, yerr=1.96 * se_dwb,
    #     fmt='o', markersize=error_marker_size, 
    #     linewidth=ci_linewidth,
    #     capsize=error_capsize, 
    #     # linestyle='-',
    #     # label="BRNEvDWB $g$ Mean ± 95% CI",
    #     color='magenta'
    # )
    # error_container2 = ax.errorbar(
    #     b_3_bh, g_human, yerr=1.96 * se_human,
    #     fmt='o', markersize=error_marker_size, 
    #     linewidth=ci_linewidth,
    #     capsize=error_capsize, 
    #     # linestyle='-',
    #     # label="BRNEvHuman $g$ Mean ± 95% CI",
    #     color='red'
    # )
    # error_container3 = ax.errorbar(
    #     b_4_bt, g_teleop, yerr=1.96 * se_teleop,
    #     fmt='o', markersize=error_marker_size, 
    #     linewidth=ci_linewidth,
    #     capsize=error_capsize, 
    #     # linestyle='-',
    #     # label="BRNEvTeleop $g$ Mean ± 95% CI",
    #     color='green'
    # )

    # We'll collect these for the legend
    error_handles = [error_container1, error_container2, error_container3]

    # 3) Optionally add line fits + threshold lines if if_g_line_fit == 'True'
    regression_handles = []
    threshold_handles = []

    if if_g_line_fit == 'True':


        # --- IDLAB---
        if p_lin_g_brne_dwb < 0.05:
            label_dwb = (f"BRNEvDWB nominal$={round(i_g_brne_dwb,2)}$, "
                         f"$\\Delta_{{gap}}={round(s_g_brne_dwb, 2)}"
                         f"\\Delta g/\\Delta \\rho$, "
                         f"$r={r_g_brne_dwb}$")
            x_line = np.linspace(cutoff_left, b_2[-1], 200)
            y_line = i_g_brne_dwb + s_g_brne_dwb * x_line
            # linebrnedwb, = ax.plot(
            #     x_line, y_line,
            #     '', linewidth=regression_linewidth,
            #     color='magenta', clip_on=True,
            #     label=label_dwb
            # )
            linebrnedwb, = ax.plot(
                [], [],
                '', linewidth=regression_linewidth,
                color='magenta', clip_on=True,
                label=label_dwb
            )
        elif p_g_spear_brne_dwb < 0.05:
            linebrnedwb, = plt.plot([], [], 'k--',
                label=(f"$g$ BRNEvDWB monotonic correlation "
                      f"$r_s={round(r_g_spear_brne_dwb, 2)}") ,
                linewidth=regression_linewidth,
                color='magenta'
            )
        else:
            linebrnedwb, = plt.plot([], [], 'k--',
                label="$g$ BRNEvDWB no correlation",
                linewidth=regression_linewidth,
                color='magenta'
            )

        regression_handles.append(linebrnedwb)
        # PRINT REGRESSION PARAMETERS BRNEVDWB
        print(f"SIM2REAL GAP {state_2} "
            f"p_l^g={p_lin_g_brne_dwb:.1f} "
            f"beta_0^g={i_g_brne_dwb:.1f}, "
            f"d_gap/d_rho={s_g_brne_dwb:.1f}, "
            f"r_l^g={r_g_brne_dwb:.1f}, "
            f"p_m^g={p_g_spear_brne_dwb:.1f}"
            f"r_m^g={r_g_spear_brne_dwb:.1f}, ")



        # --- SFM ---
        if p_lin_g_brne_human < 0.05:
            label_human = (f"BRNEvHuman nominal={round(i_g_brne_human ,2)}, "
                           f"$\\Delta_{{gap}}={round(s_g_brne_human, 2)}"
                           f"\\Delta g/\\Delta \\rho$, "
                           f"$r={r_g_brne_human}$")
            x_line = np.linspace(cutoff_left, b_3[-1], 200)
            y_line = i_g_brne_human + s_g_brne_human * x_line
            # linebrnehuman, = ax.plot(
            #     x_line, y_line,
            #     '--', linewidth=regression_linewidth,
            #     color='red', clip_on=True,
            #     label=label_human
            # )
            linebrnehuman, = ax.plot(
                [], [],
                '--', linewidth=regression_linewidth,
                color='red', clip_on=True,
                label=label_human
            )
        elif p_g_spear_brne_human < 0.05:
            linebrnehuman, = plt.plot([], [], 'k--',
                label=(f"$g$ BRNEvHuman monotonic correlation"
                       f"$r_s={round(r_g_spear_brne_human, 2)}$"),
                linewidth=regression_linewidth,
                color='red'
            )
        else:
            linebrnehuman, = plt.plot([], [], 'k--',
                label="$g$ BRNEvHuman no correlation",
                linewidth=regression_linewidth,
                color='red'
            )
        regression_handles.append(linebrnehuman)
        # PRINT REGRESSION PARAMETERS 
        print(f"SIM2REAL GAP {state_3} "
            f"p_l^g={p_lin_g_brne_human:.1f} "
            f"beta_0={i_g_brne_human:.1f}, "
            f"d_gap/d_rho={s_g_brne_human:.1f}, "
            f"r_l^g={r_g_brne_human:.1f}, "
            f"p_m^g={p_g_spear_brne_human:.1f}"
            f"r_m^g={r_g_spear_brne_human:.1f}, ")


        # --- ORCA ---
        if p_lin_g_brne_teleop < 0.05:
            label_teleop = (f"BRNEvTeleop nominal={round(i_g_brne_teleop,2)}, "
                         f"$\\Delta_{{gap}}={round(s_g_brne_teleop, 2)}"
                         f"\\Delta g/\\Delta \\rho$, "
                         f"$r={r_g_brne_teleop}$")
            # linebrneteleop, = plt.plot(
            #     b_3[:index_bt],
            #     i_g_brne_teleop + s_g_brne_teleop * b_3[:index_bt],
            #     'k--',
            #     linewidth=regression_linewidth,
            #     label=label_teleop,
            #     color='green'
            # )
            x_line = np.linspace(cutoff_left, b_4[-1], 200)
            y_line = i_g_brne_teleop + s_g_brne_teleop * x_line
            linebrneteleop, = ax.plot(
                x_line, y_line,
                '--', linewidth=regression_linewidth,
                color='green', clip_on=True,
                label=label_teleop
            )
        elif p_g_spear_brne_teleop < 0.05:
            linebrneteleop, = plt.plot([], [], 'k--',
                label=(f"$g$ BRNEvTeleop monotonic correlation "
                f"$r={round(r_g_spear_brne_teleop, 2)}$"),
                linewidth=regression_linewidth,
                color='green'
            )
        else:
            linebrneteleop, = plt.plot([], [], 'k--',
                label="$g$ BRNEvTeleop no correlation",
                linewidth=regression_linewidth,
                color='green'
            )
        regression_handles.append(linebrneteleop)
        # PRINT REGRESSION PARAMETERS BRNEVTELEOP
        print(f"SIM2REAL GAP {state_4} "
            f"p_l^g={p_lin_g_brne_teleop:.1f} "
            f"beta_0={i_g_brne_teleop:.1f}, "
            f"d_gap/d_rho={s_g_brne_teleop:.1f}, "
            f"r_l^g={r_g_brne_teleop:.1f}, "
            f"p_m^g={p_g_spear_brne_teleop:.1f}"
            f"r_m^g={r_g_spear_brne_teleop:.1f}, ")

        # --- Threshold lines ---
        # thresholdupper, = ax.plot(
        #     [0, lim_x], [0.5, 0.5], color='k', linestyle='-', lw=3.0,
        #     label='$g>0.5$ Sim2real gap not closed'
        # )
        # thresholdlower, = ax.plot(
        #     [0, lim_x], [-0.5, -0.5], color='k', linestyle='-', lw=3.0,
        #     label='$g<-0.5$ Sim2real gap not closed'
        # )


        thresholdupper, = ax.plot(
            [cutoff_left, bin_clip],
            [0.5, 0.5],
            color='k', linestyle='-', lw=3.0,
            label='$g>0.5$ Sim2real gap not closed'
        )
        thresholdlower, = ax.plot(
            [cutoff_left, bin_clip],
            [-0.5, -0.5],
            color='k', linestyle='-', lw=3.0,
            label='$g<-0.5$ Sim2real gap not closed'
        )

        threshold_handles.extend([thresholdupper, thresholdlower])




        # ERROR (CI), REGRESSION, AND THRESHOLD HANDLES
        # handles = error_handles + regression_handles + threshold_handles
        # bin_label = f'BRNEv{{}} Performance Gap' #FOR REGRESSION HANDLES
        # labels = [
        #     bin_label.format(name_2),
        #     bin_label.format(name_3),
        #     bin_label.format(name_4)] \
        #       + [h.get_label() for h in regression_handles] + \
        #               [thresholdupper.get_label(), thresholdlower.get_label()]
        # JUST THE REGRESSION AND THRESHOLD HANDLES; NO CI
        # handles = regression_handles + threshold_handles
        # bin_label = f'BRNEv{{}} Performance Gap' #FOR REGRESSION HANDLES
        # # The first three labels come from the error containers
        # labels = [h.get_label() for h in regression_handles] + \
        #                 [thresholdupper.get_label(), thresholdlower.get_label()]
        #
        handles = error_handles + threshold_handles
        bin_label = f'{{}} Sim2real Gap' #FOR CI HANDLES
        labels = [
            bin_label.format(state_2),
            bin_label.format(state_3),
            bin_label.format(state_4)] \
              + \
                      [thresholdupper.get_label(), thresholdlower.get_label()]

    else: # IF NO REGRESSION IS CONDUCTED
        thresholdupper, = ax.plot(
            [0, lim_x], [0.5, 0.5], 
            color='k',
            linestyle='-', 
            linewidth=regression_linewidth,
            label='Sim2real threshold'
        )
        thresholdlower, = ax.plot(
            [0, lim_x], [-0.5, -0.5], 
            color='k',
            linestyle='-', 
            linewidth=regression_linewidth,
            # label='$g<-0.5$ baseline outperforms BRNE'
        )
        threshold_handles.extend([thresholdupper, thresholdlower])

        bin_label = 'Sim2Real Gap of {}'
        handles = error_handles + threshold_handles
        labels = [
            bin_label.format(state_2),
            bin_label.format(state_3),
            bin_label.format(state_4),
            thresholdupper.get_label(),
            thresholdlower.get_label()
        ]

    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=legend_fontsize,
        handler_map={ErrorbarContainer: HandlerErrorbar()},
        loc=legend_location
    )

    ax.set_title(f"{performance_gap_title} for {name_1} at {site_1}", fontsize=fontsize)
    if "max" in iv:
        ax.set_xlabel("Max Crowd Density (people$/m^2$)", fontsize=fontsize)
    else:
        ax.set_xlabel("Mean Crowd Density (people$/m^2$)", fontsize=fontsize)
    ax.set_ylabel(f"{metric_1} Sim2Real Gap", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=xy_label_size)
    ax.set_xlim(cutoff_left, cutoff_right)























