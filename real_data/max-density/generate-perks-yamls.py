import zipfile
import io

# Define YAML file contents for PERKS
yamls = {
    "real_perks_BRNE_min_dist_to_person_max.yaml": """\
site: PERKS
state: Real
baseline: BRNE
metric: Min Dist To Person
mean: [3.757, 1.147, 0.905, 0.724, 0.604, 0.447, 0.488]
std: [1.983, 0.283, 0.255, 0.233, 0.214, 0.109, 0.095]
N: [3, 48, 117, 70, 25, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_DWB_min_dist_to_person_max.yaml": """\
site: PERKS
state: Real
baseline: DWB
metric: Min Dist To Person
mean: [1.072, 0.838, 0.687, 0.654, 0.654, 0.377]
std: [0.253, 0.264, 0.227, 0.209, 0.0, 0.0]
N: [6, 24, 18, 7, 1, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_Teleop_min_dist_to_person_max.yaml": """\
site: PERKS
state: Real
baseline: Teleop
metric: Min Dist To Person
mean: [1.622, 0.749]
std: [0.0, 0.207]
N: [1, 7]
bin: [0.25, 0.55]
""",
    "real_perks_BRNE_ave_safety_dist_max.yaml": """\
site: PERKS
state: Real
baseline: BRNE
metric: Ave Safety Dist
mean: [1.427, 1.382, 1.332, 1.222, 1.182, 1.157]
std: [0.186, 0.164, 0.125, 0.148, 0.119, 0.007]
N: [48, 117, 70, 25, 8, 2]
bin: [0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_DWB_ave_safety_dist_max.yaml": """\
site: PERKS
state: Real
baseline: DWB
metric: Ave Safety Dist
mean: [1.303, 1.27, 1.231, 1.241, 0.996, 0.975]
std: [0.237, 0.16, 0.133, 0.11, 0.0, 0.0]
N: [6, 24, 18, 7, 1, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_Teleop_ave_safety_dist_max.yaml": """\
site: PERKS
state: Real
baseline: Teleop
metric: Ave Safety Dist
mean: [1.678, 1.334]
std: [0.0, 0.09]
N: [1, 7]
bin: [0.25, 0.55]
""",
    "real_perks_BRNE_efficiency_max.yaml": """\
site: PERKS
state: Real
baseline: BRNE
metric: Efficiency
mean: [0.952, 0.961, 0.929, 0.948, 0.943, 0.831, 0.937]
std: [0.03, 0.057, 0.137, 0.072, 0.032, 0.307, 0.021]
N: [2, 47, 117, 70, 25, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_DWB_efficiency_max.yaml": """\
site: PERKS
state: Real
baseline: DWB
metric: Efficiency
mean: [0.969, 0.971, 0.877, 0.781, 0.997, 0.852]
std: [0.051, 0.058, 0.214, 0.327, 0.0, 0.0]
N: [6, 24, 18, 7, 1, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_Teleop_efficiency_max.yaml": """\
site: PERKS
state: Real
baseline: Teleop
metric: Efficiency
mean: [1.006, 1.001]
std: [0.0, 0.004]
N: [1, 7]
bin: [0.25, 0.55]
""",
    "real_perks_BRNE_velocity_max.yaml": """\
site: PERKS
state: Real
baseline: BRNE
metric: Velocity
mean: [0.375, 0.518, 0.497, 0.489, 0.401, 0.351, 0.388]
std: [0.265, 0.097, 0.091, 0.06, 0.121, 0.083, 0.048]
N: [3, 48, 117, 70, 25, 8, 2]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_DWB_velocity_max.yaml": """\
site: PERKS
state: Real
baseline: DWB
metric: Velocity
mean: [0.421, 0.418, 0.383, 0.356, 0.448, 0.293]
std: [0.045, 0.047, 0.055, 0.123, 0.0, 0.0]
N: [6, 24, 18, 7, 1, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35, 1.65]
""",
    "real_perks_Teleop_velocity_max.yaml": """\
site: PERKS
state: Real
baseline: Teleop
metric: Velocity
mean: [0.572, 0.411]
std: [0.0, 0.035]
N: [1, 7]
bin: [0.25, 0.55]
"""
}

# Create a BytesIO stream to hold the zip file in memory
zip_buffer = io.BytesIO()

# Write each YAML file into the zip file
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename, content in yamls.items():
        zf.writestr(filename, content)

# Save the zip file to disk so it can be downloaded
zip_filename = "real_perks_yamls.zip"
with open(zip_filename, "wb") as f:
    f.write(zip_buffer.getvalue())

print(f"Created {zip_filename}")
