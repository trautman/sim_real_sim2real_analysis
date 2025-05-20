
# FROM max-independent-variable folder run python ../../compare-4-baseline-performance.py real_arcade_*_ave_safety_dist_max.yaml
#~Dropbox/work/projects/harmonious-navigation/2024/papers/field-study/analysis-code/sim_real_sim2real_analysis/real_data/max-independent-variable
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml
import sys

from  compare_baselines_support_functions\
				import calling_data_files, hedges_g, lin_regress, ttest_p_distribution, \
				error_and_line_fit_baseline, error_and_line_fit_g

# BRNE 1 DWB 2 human 3 teleop 4
if __name__ == "__main__":

	# IMPORT DATA
	site_1, state_1, name_1, metric_1, m_1, std_1, b_1, N_1, \
	site_2, state_2, name_2, metric_2, m_2, std_2, b_2, N_2, \
	site_3, state_3, name_3, metric_3, m_3, std_3, b_3, N_3, \
	site_4, state_4, name_4, metric_4, m_4, std_4, b_4, N_4, \
	m_1_true, std_1_true, b_1_true, N_1_true, m_2_true, std_2_true, \
	b_2_true, N_2_true, m_3_true, std_3_true, b_3_true, N_3_true, m_4_true, \
	std_4_true, b_4_true, N_4_true = \
				calling_data_files(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


	# =====================
	# STATISTICAL ANALYSIS
	# COMPUTE HEDGES' G
	total_wins = 0
	total_comparisons = 0
	index_bd = min(len(m_1_true), len(m_2_true))
	g_brne_dwb, se_g_brne_dwb, z_value_brne_dwb, \
				ci_upper_brne_dwb, ci_lower_brne_dwb = \
						hedges_g(m_1_true[:index_bd], m_2_true[:index_bd], \
										 std_1_true[:index_bd], std_2_true[:index_bd], \
										 N_1_true[:index_bd], N_2_true[:index_bd], name_2)
	print('{} brne beats DWB {}/{} times, g={}'.format(\
							metric_1, np.sum(g_brne_dwb>0), len(g_brne_dwb), g_brne_dwb))
	total_wins += np.sum(g_brne_dwb>0)
	total_comparisons += len(g_brne_dwb)

	index_bh = min(len(m_1_true), len(m_3_true))
	g_brne_human, se_g_brne_human, z_value_brne_human, \
							ci_upper_brne_human, ci_lower_brne_human = \
						hedges_g(m_1_true[:index_bh], m_3_true[:index_bh], \
										 std_1_true[:index_bh], std_3_true[:index_bh], \
										 N_1_true[:index_bh], N_3_true[:index_bh], name_3)
	print('brne beats HUMAN {}/{} times, g={}'.format(\
							np.sum(g_brne_human>0), len(g_brne_human), g_brne_human))
	total_wins += np.sum(g_brne_human>0)
	total_comparisons += len(g_brne_human)

	index_bt = min(len(m_1_true), len(m_4_true))
	g_brne_teleop, se_g_brne_teleop, z_value_brne_teleop, \
							ci_upper_brne_teleop, ci_lower_brne_teleop = \
					hedges_g(m_1_true[:index_bt], m_4_true[:index_bt], \
									 std_1_true[:index_bt], std_4_true[:index_bt], \
									 N_1_true[:index_bt], N_4_true[:index_bt], name_4)
	print('brne beats TELEOP {}/{} times, g={}'.format(\
							np.sum(g_brne_teleop>0), len(g_brne_teleop), g_brne_teleop))
	total_wins += np.sum(g_brne_teleop>0)
	total_comparisons += len(g_brne_teleop)


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

	num_trials = 100
	p_value_dist_bd = ttest_p_distribution(m_1_true, m_2_true, std_1_true, \
																	 std_2_true, N_1_true, N_2_true, num_trials)
	# print('actual Pvalues', np.round(p_value_dist_bd[0],2))
	stat_sig_count = 0
	print('BRNEvDWB p_value estimation {} {}'.format(metric_1, site_1))
	for i, p_values in enumerate(p_value_dist_bd):
	  # print(f"Bin {b_1_true[i]}: {100*np.mean(np.array(p_values) < 0.05):.10g}% are stat sig, "
	  #       f"mu_p {np.mean(p_values):.2f}")
	  if 100*np.mean(np.array(p_values) < 0.05) > 80:
	  	stat_sig_count += 1

	print('BRNEVTELEOP p_value estimation {} {}'.format(metric_1, site_1))
	p_value_dist_bt = ttest_p_distribution(m_1_true, m_4_true, std_1_true, \
																	 std_4_true, N_1_true, N_4_true, num_trials)
	# print('actual Pvalues', np.round(p_value_dist_bd[0],2))
	for i, p_values in enumerate(p_value_dist_bt):
	  # print(f"Bin {b_1_true[i]}: {100*np.mean(np.array(p_values) < 0.05):.10g}% are stat sig, "
	  #       f"mu_p={np.mean(p_values):.2f}")
	  if 100*np.mean(np.array(p_values) < 0.05) > 80:
	  	stat_sig_count += 1

	print('BRNEVHUMAN p_value estimation {} {}'.format(metric_1, site_1))
	p_value_dist_bh = ttest_p_distribution(m_1_true, m_3_true, std_1_true, \
															 		std_3_true, N_1_true, N_3_true, num_trials)
	# print('actual Pvalues', np.round(p_value_dist_bd[0],2))
	for i, p_values in enumerate(p_value_dist_bh):
	  # print(f"Bin {b_1_true[i]}: {100*np.mean(np.array(p_values) < 0.05):.10g}% are stat sig, "
	  #       f"mu_p={np.mean(p_values):.2f}")
	  if 100*np.mean(np.array(p_values) < 0.05) > 80:
	  	stat_sig_count += 1

	# print('NUMBER OF STATISTICALLY SIGNIFICANT BINS', stat_sig_count)
	# print('total wins {} total runs {}'.format(total_wins, total_comparisons))
	print('wins/#runs={}/{}, stat sig={}'.format(total_wins, total_comparisons, stat_sig_count))

	# # COMPUTE LINEAR REGRESS OF BASELINES AND G
	s_1, s_2, s_3, s_4, s_g_brne_dwb, s_g_brne_human, s_g_brne_teleop, \
	i_1, i_2, i_3, i_4, i_g_brne_dwb, i_g_brne_human, i_g_brne_teleop, \
	r_1, r_2, r_3, r_4, r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop, \
	p_lin_1, p_lin_2, p_lin_3, p_lin_4, p_lin_g_brne_dwb, p_lin_g_brne_human, \
	p_lin_g_brne_teleop,  r_spear_1, r_spear_2, r_spear_3, r_spear_4, \
	p_spear_1, p_spear_2, p_spear_3, p_spear_4, r_g_spear_brne_dwb, \
	p_g_spear_brne_dwb, r_g_spear_brne_human, p_g_spear_brne_human, \
  r_g_spear_brne_teleop, p_g_spear_brne_teleop = \
		lin_regress(name_1, name_2, name_3, name_4, metric_1, metric_2, metric_3, \
			metric_4, b_1_true, b_2_true, b_3_true, b_4_true, m_1_true, m_2_true, \
			m_3_true, m_4_true, g_brne_dwb, g_brne_human, g_brne_teleop, index_bd, \
			index_bh, index_bt, site_1, site_2, site_3, site_4)


	# # ======================
	# # PLOT BASELINES
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
	performance_title = "Performance Versus Density"
	error_and_line_fit_baseline(name_1, name_2, name_3, name_4, site_1, site_2, \
		site_3, site_4, state_1, state_2, state_3, state_4, metric_1, metric_2, \
		metric_3, metric_4, b_1_true, b_2_true, b_3_true, b_4_true, m_1_true, \
		m_2_true, m_3_true, m_4_true, std_1_true, std_2_true, std_3_true, \
		std_4_true, s_1, s_2, s_3, s_4, i_1, i_2, i_3, i_4, r_1, r_2, r_3, r_4, \
		p_lin_1, p_lin_2, p_lin_3, p_lin_4, N_1_true, N_2_true, N_3_true, N_4_true, \
		axb, r_spear_1, r_spear_2, r_spear_3, r_spear_4, p_spear_1, p_spear_2, \
		p_spear_3, p_spear_4, error_fontsize, error_marker_size, \
		regression_linewidth, \
		error_capsize, legend_fontsize, correlation_threshold, performance_title, \
		ci_linewidth, legend_location_perf, xy_label_size, sys.argv[1])

	# # PLOT g
	fig2, axg = plt.subplots(figsize=figure_size)
	lim_x_index = max(index_bd, index_bh, index_bt)
	print("lim_x_index", lim_x_index)
	print("bd {} bh {} bt {}".format(index_bd, index_bh, index_bt))
	if "max" in sys.argv[1]:
		lim_x = 0.275*lim_x_index - 0.275
	else:
		lim_x = 0.05*lim_x_index - 0.05
	if_g_line_fit = 'True'
	correlation_threshold = 0.5
	legend_location_g = 'best'
	ci_threshold = 50
	performance_gap_title = "Performance Gap Versus Density"
	error_and_line_fit_g(name_1, name_2, name_3, name_4, site_1, site_2, site_3, \
		site_4, metric_1, metric_2, metric_3, metric_4, b_1_true, b_2_true, \
		b_3_true, b_4_true, g_brne_dwb, g_brne_human, g_brne_teleop, \
		se_g_brne_dwb, \
		se_g_brne_human, se_g_brne_teleop, s_g_brne_dwb, s_g_brne_human, \
		s_g_brne_teleop, i_g_brne_dwb, i_g_brne_human, i_g_brne_teleop, \
		r_g_brne_dwb, r_g_brne_human, r_g_brne_teleop, p_lin_g_brne_dwb, \
		p_lin_g_brne_human, p_lin_g_brne_teleop, axg, index_bd, index_bh, \
		index_bt, \
		r_g_spear_brne_dwb, p_g_spear_brne_dwb, r_g_spear_brne_human, \
		p_g_spear_brne_human, r_g_spear_brne_teleop, p_g_spear_brne_teleop, \
	  error_fontsize, error_marker_size, regression_linewidth, error_capsize, \
	  legend_fontsize, lim_x, if_g_line_fit, correlation_threshold, \
	  performance_gap_title, ci_threshold, ci_linewidth, legend_location_g, \
	  xy_label_size, sys.argv[1])

	fig1.tight_layout()
	fig2.tight_layout()
	plt.show()



# LEGEND LOCATIONS
# best
# 	upper right
# 	upper left
# 	lower left
# 	lower right
# 	right
# 	center left
# 	center right
# 	lower center
# 	upper center
# 	center