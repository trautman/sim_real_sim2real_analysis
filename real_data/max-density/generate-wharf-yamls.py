import zipfile
import io

# Define YAML file contents for WHARF
yamls = {
    # Metric: Min Dist To Person
    "real_wharf_BRNE_min_dist_to_person_max.yaml": """\
site: WHARF
state: Real
baseline: BRNE
metric: Min Dist To Person
mean: [2.568, 1.426, 0.985, 0.832, 0.596, 0.43]
std: [0.492, 0.362, 0.298, 0.225, 0.31, 0.074]
N: [25, 87, 75, 29, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_wharf_DWB_min_dist_to_person_max.yaml": """\
site: WHARF
state: Real
baseline: DWB
metric: Min Dist To Person
mean: [2.243, 1.298, 0.936, 0.753, 0.762, 0.464, 0.198]
std: [0.937, 0.349, 0.303, 0.238, 0.205, 0.09, 0.0]
N: [33, 83, 81, 20, 6, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.9]
""",
    "real_wharf_Teleop_min_dist_to_person_max.yaml": """\
site: WHARF
state: Real
baseline: Teleop
metric: Min Dist To Person
mean: [2.518, 1.389, 0.937, 0.761, 0.905, 0.515]
std: [0.441, 0.328, 0.294, 0.294, 0.219, 0.0]
N: [30, 71, 46, 18, 3, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.65]
""",
    # Metric: Ave Safety Dist
    "real_wharf_BRNE_ave_safety_dist_max.yaml": """\
site: WHARF
state: Real
baseline: BRNE
metric: Ave Safety Dist
mean: [1.557, 1.392, 1.318, 1.246, 1.135]
std: [0.28, 0.193, 0.157, 0.163, 0.075]
N: [87, 75, 29, 8, 2]
bin: [0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_wharf_DWB_ave_safety_dist_max.yaml": """\
site: WHARF
state: Real
baseline: DWB
metric: Ave Safety Dist
mean: [1.302, 1.429, 1.35, 1.284, 1.274, 1.046, 0.976]
std: [0.146, 0.282, 0.23, 0.153, 0.183, 0.024, 0.0]
N: [9, 83, 81, 20, 6, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.9]
""",
    "real_wharf_Teleop_ave_safety_dist_max.yaml": """\
site: WHARF
state: Real
baseline: Teleop
metric: Ave Safety Dist
mean: [1.502, 1.4, 1.271, 1.376, 1.197]
std: [0.3, 0.188, 0.223, 0.158, 0.0]
N: [71, 46, 18, 3, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.65]
""",
    # Metric: Efficiency
    "real_wharf_BRNE_efficiency_max.yaml": """\
site: WHARF
state: Real
baseline: BRNE
metric: Efficiency
mean: [0.999, 0.987, 0.975, 0.957, 0.909, 0.943]
std: [0.002, 0.013, 0.02, 0.037, 0.074, 0.011]
N: [25, 87, 75, 29, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_wharf_DWB_efficiency_max.yaml": """\
site: WHARF
state: Real
baseline: DWB
metric: Efficiency
mean: [0.988, 0.964, 0.945, 0.935, 0.911, 0.816, 0.939]
std: [0.018, 0.101, 0.136, 0.143, 0.061, 0.171, 0.0]
N: [33, 83, 81, 20, 6, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.9]
""",
    "real_wharf_Teleop_efficiency_max.yaml": """\
site: WHARF
state: Real
baseline: Teleop
metric: Efficiency
mean: [0.982, 0.961, 0.93, 0.951, 0.935, 0.278]
std: [0.012, 0.049, 0.116, 0.047, 0.046, 0.0]
N: [30, 71, 46, 18, 3, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.65]
""",
    # Metric: Velocity
    "real_wharf_BRNE_velocity_max.yaml": """\
site: WHARF
state: Real
baseline: BRNE
metric: Velocity
mean: [0.527, 0.524, 0.503, 0.479, 0.391, 0.152]
std: [0.038, 0.024, 0.042, 0.074, 0.102, 0.043]
N: [25, 87, 75, 29, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_wharf_DWB_velocity_max.yaml": """\
site: WHARF
state: Real
baseline: DWB
metric: Velocity
mean: [0.519, 0.505, 0.493, 0.482, 0.488, 0.389, 0.064]
std: [0.013, 0.026, 0.037, 0.052, 0.016, 0.026, 0.0]
N: [33, 83, 81, 20, 6, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.9]
""",
    "real_wharf_Teleop_velocity_max.yaml": """\
site: WHARF
state: Real
baseline: Teleop
metric: Velocity
mean: [0.617, 0.611, 0.574, 0.548, 0.657, 0.304]
std: [0.055, 0.05, 0.062, 0.09, 0.027, 0.0]
N: [30, 71, 46, 18, 3, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.65]
"""
}

# Create a BytesIO stream to hold the zip file in memory
zip_buffer = io.BytesIO()

# Write each YAML file into the zip file
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename, content in yamls.items():
        zf.writestr(filename, content)

# Write the zip file to disk so it can be downloaded
zip_filename = "real_wharf_yamls.zip"
with open(zip_filename, "wb") as f:
    f.write(zip_buffer.getvalue())

print(f"Created {zip_filename}")
