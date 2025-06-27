A command-line toolkit for comparing simulated and real-world performance of robotic and pedestrian navigation systems. Includes tools for evaluating performance metrics, plotting baseline comparisons, and quantifying sim2real gaps.


## 🚀 Usage


### 📊 Pedestrian/Robot Sim2Real Comparison
The command
```bash
python compare_pedestrian_sim2real.py --label ped --no-display <REAL_YAML> <SIM1_YAML> <SIM2_YAML> <SIM3_YAML>
```
compares real Santa Cruz pedestrians with pedestrians simulated using IDLab, ORCA, and SFM. 

For example
```bash
python compare_pedestrian_sim2real.py --label ped human-data/real_arcade_BRNE_average_safety_distance_mean.yaml human-data/sim_arcade_BRNE_IDLab_average_safety_distance_mean.yaml human-data/sim_arcade_BRNE_ORCA_average_safety_distance_mean.yaml human-data/sim_arcade_BRNE_SFM_average_safety_distance_mean.yaml
```
which will create and display two figures in the pwd, namely  
####Sample output
*asd_arcade_ped_brne_idlab_orca_sfm_gap.png
*asd_arcade_ped_brne_idlab_orca_sfm_perf.png




The command
```bash
python compare_robot_sim2real.py --label robot --no-display <REAL_YAML> <SIM1_YAML> <SIM2_YAML> <SIM3_YAML>
```
compares real robot performance (e.g. robot in real world) compared to robot performance in IDLab-, ORCA-, or SFM-based pedestrian simulation environments. 



---


### 🔁 Batch Compare (All YAMLs in folder)

```bash
python batch_compare.py --label ped <input_yaml_dir> <output_fig_dir>
python batch_compare.py --label robot <input_yaml_dir> <output_fig_dir>
```

* Runs `compare_pedestrian_sim2real.py` or `compare_robot_sim2real.py`
* Auto-pairs `real_*.yaml` with matching `sim_*.yaml`
* Outputs figures comparing average safety distance, min distance to person, path efficiency, total time, and translational velocity, organized by density bin, with mean +/- 95% CI in each bin, linear regression; second plot compares gap between BRNE and other robot algorithms and between sim2real gap of real world brne performance and 3 different simulators (ORCA, IDLab, SFM).

---

### 📉 Baseline Comparison (4-way)

```bash
python compare_baselines.py real_arcade_BRNE_<METRIC>.yaml real_arcade_DWB_<METRIC>.yaml real_arcade_human_<METRIC>.yaml real_arcade_teleop_<METRIC>.yaml
```

* Works with both `mean-density` and `max-density`

---

### 🏞️ Site Comparison (3-site)

```bash
python compare_sites.py real_<site1>_BRNE_<METRIC>.yaml real_<site2>_BRNE_<METRIC>.yaml real_<site3>_BRNE_<METRIC>.yaml
```

---

### ⚖️ One-to-One Sim2Real

```bash
python compare_sim2real.py sim_data/<SIM_YAML> real_data/<REAL_YAML>
```

---

## 📷 Example Output

### Performance Plot

![Performance](docs/example_perf.png)

### Gap Plot

![Gap](docs/example_gap.png)

---

## 🧠 Metric Abbreviations

| Full Name                      | Abbreviation |
| ------------------------------ | ------------ |
| `average_safety_distance_mean` | `asd`        |
| `min_distance_to_person`       | `mdp`        |

---

## ✅ Output

Each script may output:

* `*_perf.png` (Performance vs. Density)
* `*_gap.png` (Performance Gap vs. Density)
* Printed statistics: Hedges' g, regression slopes, t-tests

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

Developed for sim2real evaluation in crowded navigation environments (2024–2025 field study). Contributions welcome!
