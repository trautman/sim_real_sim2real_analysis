
# Sim2Real Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

A command-line toolkit for comparing simulated and real-world performance of robotic and pedestrian navigation systems. Includes tools for evaluating performance metrics, plotting baseline comparisons, and quantifying sim2real gaps.

---

## 📦 Installation

1. Clone this repository:

```bash
git clone https://github.com/your-org/sim2real-analysis.git
cd sim2real-analysis
```

2. Create a Python environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

> Requires Python >= 3.8

---

## 📁 Folder Structure

```
sim2real-analysis/
├── batch_compare.py
├── compare_baselines.py
├── compare_sites.py
├── compare_sim2real.py
├── compare_pedestrian_sim2real.py
├── compare_robot_sim2real.py
├── *_support_functions.py
├── real_data/
│   ├── mean-density/
│   └── max-density/
├── sim_data/
└── output_figures/
```

---

## 🚀 Usage

### 🔁 Batch Compare (All YAMLs in folder)

```bash
python batch_compare.py --label ped <input_yaml_dir> <output_fig_dir>
python batch_compare.py --label robot <input_yaml_dir> <output_fig_dir>
```

* Runs `compare_pedestrian_sim2real.py` or `compare_robot_sim2real.py`
* Auto-pairs `real_*.yaml` with matching `sim_*.yaml`

---

### 📊 Pedestrian/Robot Sim2Real Comparison

```bash
python compare_pedestrian_sim2real.py --label ped --no-display <REAL_YAML> <SIM1_YAML> <SIM2_YAML> <SIM3_YAML>
python compare_robot_sim2real.py --label robot --no-display <REAL_YAML> <SIM1_YAML> <SIM2_YAML> <SIM3_YAML>
```

#### Output

* `mdp_arcade_ped_brne_idlab_sfm_orca_perf.png`
* `mdp_arcade_ped_brne_idlab_sfm_orca_gap.png`

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
