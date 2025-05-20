import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml
import sys

from compare_sim2real_support_functions \
    import calling_data_files, hedges_g, ttest_p_distribution, lin_regress, \
            error_and_line_fit_baseline, error_and_line_fit_g

# from real_baseline_statistical_analysis_support_functions\
#                 import calling_data_files, hedges_g, lin_regress, ttest_p_distribution, \
#                 error_and_line_fit_baseline, error_and_line_fit_g

# BRNE 1 DWB 2 human 3 teleop 4
if __name__ == "__main__":

# SIM DATA IS SYS.ARGV[1]
# REAL DATA IS SYS.ARGV[2]
    # IMPORT DATA;
    site_1, state_1, name_1, metric_1, m_1, std_1, b_1, N_1, \
    site_2, state_2, name_2, metric_2, m_2, std_2, b_2, N_2, \
    m_1_true, std_1_true, b_1_true, N_1_true, m_2_true, std_2_true, \
    b_2_true, N_2_true = \
                calling_data_files(sys.argv[1], sys.argv[2])



    # =====================
    # STATISTICAL ANALYSIS
    # COMPUTE HEDGES' G
    index_s2r = min(len(m_1_true), len(m_2_true))
    
    g_s2r, se_s2r, z_value_s2r, ci_upper_s2r, ci_lower_s2r = \
                        hedges_g(m_1_true[:index_s2r], m_2_true[:index_s2r], \
                                 std_1_true[:index_s2r], std_2_true[:index_s2r], \
                                 N_1_true[:index_s2r], N_2_true[:index_s2r], name_2)
    
    print(f"For {site_1}, {metric_1} {state_1}2{state_2} gap closed in "
          f"{np.sum(np.abs(g_s2r)<0.5)} out of {len(g_s2r)} bins.\n"
          f"sim2real GAP: {g_s2r}")
    total_closures = np.sum(np.abs(g_s2r)<0.5)
    total_gaps = len(g_s2r)





    num_trials = 100
    p_value_dist_s2r = ttest_p_distribution(m_1_true, m_2_true, std_1_true, \
                                                        std_2_true, N_1_true, N_2_true, num_trials)
    # print('actual Pvalues', np.round(p_value_dist_bd[0],2))
    stat_sig_count = 0
    # print('BRNEvDWB p_value estimation {} {}'.format(metric_1, site_1))
    print(f"{state_1}2{state_2} p_value estimation")
    for i, p_values in enumerate(p_value_dist_s2r):
      # print(f"Bin {b_1_true[i]}: {100*np.mean(np.array(p_values) < 0.05):.10g}% are stat sig, "
      #       f"mu_p {np.mean(p_values):.2f}")
      if 100*np.mean(np.array(p_values) < 0.05) > 80:
        stat_sig_count += 1
    print(f"closures/#gaps={total_closures}/{total_gaps}, #gaps with a stat sig size ={stat_sig_count}")
    print(f"WE WANT GAPS TO BE STATISTIALLY INSIGNIFICANT: " 
          f"EG NO STATISTICALLY SIGNIFICANT DIFFERENCE B/N SIM AND REAL. "
          f"IF STAT SIG COUNT LARGE THEN THAT MEANS NO SIM2REAL TRANSFER")


    s_1, s_2, s_g_s2r, i_1, i_2, i_g_s2r, p_lin_1, p_lin_2, p_lin_g_s2r, \
    r_lin_1, r_lin_2, r_lin_g_s2r, p_spear_1, p_spear_2, p_spear_g_s2r, \
    r_spear_1, r_spear_2, r_spear_g_s2r = \
                lin_regress(name_1, name_2, metric_1, metric_2, \
                            b_1_true, b_2_true, m_1_true, m_2_true, g_s2r, \
                            index_s2r, site_1, site_2)


  #   # # ======================
  #   # # PLOT BASELINES
    s_width = 3
    s_heigth = 3.5
    figure_size = (s_width*4, s_heigth*3)
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
    performance_title = "Sim to Real Comparison"
    error_and_line_fit_baseline(name_1, name_2, \
                                site_1, site_2, \
                                state_1, state_2, \
                                metric_1, metric_2, \
                                b_1_true, b_2_true, \
                                m_1_true, m_2_true, \
                                std_1_true, std_2_true, \
                                s_1, s_2, \
                                i_1, i_2, \
                                r_lin_1, r_lin_2, \
                                p_lin_1, p_lin_2, \
                                N_1_true, N_2_true, \
                                r_spear_1, r_spear_2, \
                                p_spear_1, p_spear_2, \
                                axb, \
                                error_fontsize, error_marker_size, \
                                regression_linewidth, \
                                error_capsize, legend_fontsize, \
                                correlation_threshold, performance_title, \
                                ci_linewidth, legend_location_perf, \
                                xy_label_size, sys.argv[1])

  #   # # PLOT g
    fig2, axg = plt.subplots(figsize=figure_size)
    # lim_x_index = max(index_bd, index_bh, index_bt)
    # print("lim_x_index", lim_x_index)
    # print("bd {} bh {} bt {}".format(index_bd, index_bh, index_bt))
    if "max" in sys.argv[1]:
        lim_x = 0.275*index_s2r - 0.275
    else:
        lim_x = 0.05*index_s2r - 0.05
    if_g_line_fit = 'True'
    correlation_threshold = 0.5
    legend_location_g = 'best'
    ci_threshold = 50
    performance_gap_title = "Performance Gap Versus Density"
    error_and_line_fit_g(
                        name_1, name_2, 
                        site_1, site_2, 
                        metric_1, metric_2, 
                        b_1_true, b_2_true, 
                        index_s2r, \
                        g_s2r, \
                        se_s2r, \
                        s_g_s2r, 
                        i_g_s2r, \
                        r_lin_g_s2r, \
                        p_lin_g_s2r, \
                        r_spear_g_s2r, \
                        p_spear_g_s2r, \
                        axg, \
                        error_fontsize, \
                        error_marker_size, \
                        regression_linewidth, \
                        error_capsize, \
                        legend_fontsize, \
                        lim_x, 
                        if_g_line_fit, \
                        correlation_threshold, \
                        performance_gap_title, \
                        ci_threshold, \
                        ci_linewidth, \
                        legend_location_g, \
                        xy_label_size, 
                        sys.argv[1])
    # error_and_line_fit_g(name_1, name_2, name_3, name_4, site_1, site_2, site_3, \
    #     site_4, metric_1, metric_2, metric_3, metric_4, b_1_true, b_2_true, \
    #     b_3_true, b_4_true, g_brne_dwb, g_brne_human, g_brne_teleop, \
    #     se_g_brne_dwb, \
    #     se_g_brne_human, se_g_brne_teleop, s_g_brne_dwb, s_g_brne_human, \
    #     s_g_brne_teleop, i_g_brne_dwb, i_g_brne_human, i_g_brne_teleop, \
    #     r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop, p_lin_g_brne_dwb, \
    #     p_lin_g_brne_human, p_lin_g_brne_teleop, axg, index_bd, index_bh, \
    #     index_bt, \
    #     r_g_spear_brne_dwb, p_g_spear_brne_dwb, r_g_spear_brne_human, \
    #     p_g_spear_brne_human, r_g_spear_brne_teleop, p_g_spear_brne_teleop, \
    #   error_fontsize, error_marker_size, regression_linewidth, error_capsize, \
    #   legend_fontsize, lim_x, if_g_line_fit, correlation_threshold, \
    #   performance_gap_title, ci_threshold, ci_linewidth, legend_location_g, \
    #   xy_label_size, sys.argv[1])

    # fig1.tight_layout()
    # fig2.tight_layout()
    plt.show()










# T TEST
    # p_ttest_brne_dwb = ttest_p_distribution(m_1_true, m_2_true, std_1_true, std_2_true, N_1_true, N_2_true)
    # print('p BRNE vs DWB TTEST', metric_1, p_ttest_brne_dwb)
    # p_ttest_brne_dwb = np.array(p_ttest_brne_dwb)
    # sig_indices_bd = np.where(p_ttest_brne_dwb <= 0.05)[0]
    # insig_indices_bd = np.where(np.logical_or(p_ttest_brne_dwb > 0.05, np.isnan(p_ttest_brne_dwb)))[0]
    # stat_sig_bins_bd = b_1_true[sig_indices_bd]
    # stat_insig_bins_bd = b_1_true[insig_indices_bd]
    # print('DISTINGUISHABLE BRNEVdwb BINS', stat_sig_bins_bd)
    # print('INDISTINGUISHABLE BRNEVdwb BINS', stat_insig_bins_bd)

    # p_ttest_brne_teleop = ttest_p_distribution(m_1_true, m_4_true, std_1_true, std_4_true, N_1_true, N_4_true)
    # print('p BRNE vs TELEOP TTEST', metric_1, p_ttest_brne_teleop)
    # p_ttest_brne_teleop = np.array(p_ttest_brne_teleop)
    # sig_indices_bt = np.where(p_ttest_brne_teleop <= 0.05)[0]
    # insig_indices_bt = np.where(np.logical_or(p_ttest_brne_teleop > 0.05, np.isnan(p_ttest_brne_teleop)))[0]
    # stat_sig_bins_bt = b_1_true[sig_indices_bt]
    # stat_insig_bins_bt = b_1_true[insig_indices_bt]
    # print('DISTINGUISHABLE BRNEVteleop BINS', stat_sig_bins_bt)
    # print('INDISTINGUISHABLE BRNEVteleop BINS', stat_insig_bins_bt)

    # p_ttest_brne_human = ttest_p_distribution(m_1_true, m_3_true, std_1_true, std_3_true, N_1_true, N_3_true)
    # print('p BRNE vs HUMAN TTEST', metric_1, p_ttest_brne_human)
    # p_ttest_brne_human = np.array(p_ttest_brne_human)
    # sig_indices_bh = np.where(p_ttest_brne_human <= 0.05)[0]
    # insig_indices_bh = np.where(np.logical_or(p_ttest_brne_human > 0.05, np.isnan(p_ttest_brne_human)))[0]
    # stat_sig_bins_bh = b_1_true[sig_indices_bh]
    # stat_insig_bins_bh = b_1_true[insig_indices_bh]
    # print('DISTINGUISHABLE BRNEVhuman BINS', stat_sig_bins_bh)
    # print('INDISTINGUISHABLE BRNEVhuman BINS', stat_insig_bins_bh)

# LEGEND LOCATIONS
# best
#   upper right
#   upper left
#   lower left
#   lower right
#   right
#   center left
#   center right
#   lower center
#   upper center
#   center