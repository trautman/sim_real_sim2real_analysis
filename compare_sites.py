import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml
import sys

from compare_sites_support_functions import (
    calling_data_files, hedges_g, lin_regress, ttest_p_distribution,
    error_and_line_fit_baseline, error_and_line_fit_g
)

# In the original order, assume:
# baseline 1: Reference (e.g. BRNE)
# baseline 2: Comparator 1 (e.g. DWB)
# baseline 3: Comparator 2 (e.g. HUMAN)
if __name__ == "__main__":
    # IMPORT DATA (now 3 files instead of 4)
    (site_1, state_1, name_1, metric_1, m_1, std_1, b_1, N_1,
     site_2, state_2, name_2, metric_2, m_2, std_2, b_2, N_2,
     site_3, state_3, name_3, metric_3, m_3, std_3, b_3, N_3,
     m_1_true, std_1_true, b_1_true, N_1_true,
     m_2_true, std_2_true, b_2_true, N_2_true,
     m_3_true, std_3_true, b_3_true, N_3_true) = calling_data_files(
                                            sys.argv[1], sys.argv[2], sys.argv[3])
    
    # =====================
    # STATISTICAL ANALYSIS
    # Compute Hedges' g for baseline1 vs baseline2
    index_bd = min(len(m_1_true), len(m_2_true))
    g_brne_dwb, se_g_brne_dwb, z_value_brne_dwb, ci_upper_brne_dwb, ci_lower_brne_dwb = hedges_g(
        m_1_true[:index_bd], m_2_true[:index_bd],
        std_1_true[:index_bd], std_2_true[:index_bd],
        N_1_true[:index_bd], N_2_true[:index_bd], name_2)
    print('{} (baseline1) beats {} {}/{} times, g={}'.format(
          metric_1, name_2, np.sum(g_brne_dwb > 0), len(g_brne_dwb), g_brne_dwb))
    
    # Compute Hedges' g for baseline1 vs baseline3
    index_bh = min(len(m_1_true), len(m_3_true))
    g_brne_human, se_g_brne_human, z_value_brne_human, ci_upper_brne_human, ci_lower_brne_human = hedges_g(
        m_1_true[:index_bh], m_3_true[:index_bh],
        std_1_true[:index_bh], std_3_true[:index_bh],
        N_1_true[:index_bh], N_3_true[:index_bh], name_3)
    print('{} (baseline1) beats {} {}/{} times, g={}'.format(
          metric_1, name_3, np.sum(g_brne_human > 0), len(g_brne_human), g_brne_human))
    
    # T-TEST p-value distributions (over multiple trials)
    num_trials = 100
    p_value_dist_bd = ttest_p_distribution(
        m_1_true, m_2_true, std_1_true, std_2_true, N_1_true, N_2_true, num_trials)
    stat_sig_count = 0
    print('Baseline1 vs {} p_value estimation:'.format(name_2))
    for i, p_values in enumerate(p_value_dist_bd):
        if 100 * np.mean(np.array(p_values) < 0.05) > 80:
            stat_sig_count += 1
    
    print('Baseline1 vs {} p_value estimation:'.format(name_3))
    p_value_dist_bh = ttest_p_distribution(
        m_1_true, m_3_true, std_1_true, std_3_true, N_1_true, N_3_true, num_trials)
    for i, p_values in enumerate(p_value_dist_bh):
        if 100 * np.mean(np.array(p_values) < 0.05) > 80:
            stat_sig_count += 1
    
    print('Total wins/total comparisons = {}/{}, statistically sig bins = {}'.format(
          np.sum(g_brne_dwb > 0) + np.sum(g_brne_human > 0),
          len(g_brne_dwb) + len(g_brne_human),
          stat_sig_count))
    
    # =====================
    # LINEAR REGRESSION OF BASELINES AND g
    (s_1, s_2, s_3,
     s_g_brne_dwb, s_g_brne_human,
     i_1, i_2, i_3,
     i_g_brne_dwb, i_g_brne_human,
     r_1, r_2, r_3,
     r_g_brne_dwb, r_g_brne_human,
     p_lin_1, p_lin_2, p_lin_3,
     p_lin_g_brne_dwb, p_lin_g_brne_human,
     r_spear_1, r_spear_2, r_spear_3,
     p_spear_1, p_spear_2, p_spear_3,
     r_g_spear_brne_dwb, p_g_spear_brne_dwb,
     r_g_spear_brne_human, p_g_spear_brne_human) = lin_regress(
         name_1, name_2, name_3,
         metric_1, metric_2, metric_3,
         b_1_true, b_2_true, b_3_true,
         m_1_true, m_2_true, m_3_true,
         g_brne_dwb, g_brne_human,
         index_bd, index_bh,
         site_1, site_2, site_3)
    
    # =====================
    # PLOTTING BASELINE PERFORMANCE
    s_width = 3
    s_height = 3.5
    figure_size = (s_width * 4, s_height * 3)
    fig1, axb = plt.subplots(figsize=figure_size)
    error_fontsize = 30
    error_marker_size = 13
    error_capsize = 9
    legend_fontsize = 20
    legend_location_perf = 'best'
    regression_linewidth = 5
    ci_linewidth = 3
    xy_label_size = 20
    correlation_threshold = 0.5
    performance_title = "Site Performance Versus Density"

    error_and_line_fit_baseline(
        name_1, name_2, name_3,
        site_1, site_2, site_3,
        state_1, state_2, state_3,
        metric_1, metric_2, metric_3,
        b_1_true, b_2_true, b_3_true,
        m_1_true, m_2_true, m_3_true,
        std_1_true, std_2_true, std_3_true,
        s_1, s_2, s_3,
        i_1, i_2, i_3,
        r_1, r_2, r_3,
        p_lin_1, p_lin_2, p_lin_3,
        N_1_true, N_2_true, N_3_true,
        axb,
        r_spear_1, r_spear_2, r_spear_3,
        p_spear_1, p_spear_2, p_spear_3,
        error_fontsize,
        error_marker_size,
        regression_linewidth,
        error_capsize,
        legend_fontsize,
        correlation_threshold,
        performance_title,
        ci_linewidth,
        legend_location_perf,
        xy_label_size
    )

    fig1.tight_layout()

    # =====================
    # PLOTTING PERFORMANCE GAP (g values)
    lim_x_index = max(index_bd, index_bh)
    if "max" in sys.argv[1]:
        lim_x = 0.275 * lim_x_index - 0.275
    else:
        lim_x = 0.05 * lim_x_index - 0.05
    if_g_line_fit = 'True'
    performance_gap_title = "Site Performance Gap Versus Density"

    fig2, axg = plt.subplots(figsize=(s_width * 4, s_height * 3))
    # Note: Added ci_threshold (here 50) as the first parameter after performance_gap_title.
    error_and_line_fit_g(
        name_1, name_2, name_3,
        site_1, site_2, site_3,
        metric_1, metric_2, metric_3,
        b_1_true, b_2_true, b_3_true,
        g_brne_dwb, g_brne_human,
        se_g_brne_dwb, se_g_brne_human,
        s_g_brne_dwb, s_g_brne_human,
        i_g_brne_dwb, i_g_brne_human,
        r_g_brne_dwb, r_g_brne_human,
        p_lin_g_brne_dwb, p_lin_g_brne_human,
        axg, index_bd, index_bh,
        r_g_spear_brne_dwb, p_g_spear_brne_dwb,
        r_g_spear_brne_human, p_g_spear_brne_human,
        error_fontsize,
        error_marker_size,
        regression_linewidth,
        error_capsize,
        legend_fontsize,
        lim_x,
        if_g_line_fit,
        correlation_threshold,
        performance_gap_title,
        50,                # ci_threshold added here
        ci_linewidth,
        legend_location_perf,
        xy_label_size
    )

    fig2.tight_layout()
    plt.show()
