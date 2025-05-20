import zipfile
import io

# Define YAML file contents
yamls = {
    "real_arcade_BRNE_min_dist_to_person_max.yaml": """\
site: Arcade
state: Real
baseline: BRNE
metric: Min Dist To Person
mean: [2.071, 1.217, 0.858, 0.706, 0.542, 0.499, 0.542, 0.259]
std: [0.255, 0.37, 0.283, 0.23, 0.228, 0.187, 0.075, 0.0]
N: [6, 75, 185, 126, 50, 13, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65, 1.9]
""",
    "real_arcade_DWB_min_dist_to_person_max.yaml": """\
site: Arcade
state: Real
baseline: DWB
metric: Min Dist To Person
mean: [1.279, 0.793, 0.736, 0.482, 0.451]
std: [0.413, 0.185, 0.224, 0.245, 0.0]
N: [5, 30, 18, 4, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_arcade_Teleop_min_dist_to_person_max.yaml": """\
site: Arcade
state: Real
baseline: Teleop
metric: Min Dist To Person
mean: [1.951, 1.35, 0.88, 0.75, 0.585]
std: [1.205, 0.369, 0.31, 0.229, 0.281]
N: [42, 96, 100, 39, 11]
bin: [0.0, 0.25, 0.55, 0.8, 1.1]
""",
    "real_arcade_BRNE_ave_safety_dist_max.yaml": """\
site: Arcade
state: Real
baseline: BRNE
metric: Ave Safety Dist
mean: [1.514, 1.49, 1.397, 1.32, 1.248, 1.203, 1.291, 1.164]
std: [0.0, 0.248, 0.181, 0.156, 0.149, 0.106, 0.119, 0.0]
N: [1, 75, 185, 126, 50, 13, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65, 1.9]
""",
    "real_arcade_DWB_ave_safety_dist_max.yaml": """\
site: Arcade
state: Real
baseline: DWB
metric: Ave Safety Dist
mean: [1.491, 1.324, 1.304, 1.26, 1.297]
std: [0.309, 0.175, 0.121, 0.111, 0.0]
N: [5, 30, 18, 4, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_arcade_Teleop_ave_safety_dist_max.yaml": """\
site: Arcade
state: Real
baseline: Teleop
metric: Ave Safety Dist
mean: [1.412, 1.509, 1.329, 1.307, 1.216]
std: [0.189, 0.292, 0.227, 0.179, 0.167]
N: []
bin: [0.0, 0.25, 0.55, 0.8, 1.1]
""",
    "real_arcade_BRNE_velocity_max.yaml": """\
site: Arcade
state: Real
baseline: BRNE
metric: Velocity
mean: [0.528, 0.486, 0.445, 0.417, 0.381, 0.388, 0.456, 0.205]
std: [0.099, 0.074, 0.069, 0.073, 0.069, 0.069, 0.027, 0.0]
N: [6, 75, 185, 126, 50, 13, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65, 1.9]
""",
    "real_arcade_DWB_velocity_max.yaml": """\
site: Arcade
state: Real
baseline: DWB
metric: Velocity
mean: [0.473, 0.44, 0.447, 0.325, 0.113]
std: [0.003, 0.072, 0.021, 0.134, 0.0]
N: [5, 30, 18, 4, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_arcade_Teleop_velocity_max.yaml": """\
site: Arcade
state: Real
baseline: Teleop
metric: Velocity
mean: [0.576, 0.606, 0.592, 0.565, 0.551]
std: [0.114, 0.072, 0.078, 0.097, 0.065]
N: [42, 96, 100, 39, 11]
bin: [0.0, 0.25, 0.55, 0.8, 1.1]
""",
    "real_arcade_BRNE_efficiency_max.yaml": """\
site: Arcade
state: Real
baseline: BRNE
metric: Efficiency
mean: [0.904, 0.976, 0.963, 0.962, 0.958, 0.956, 0.945, 0.967]
std: [0.171, 0.016, 0.067, 0.06, 0.051, 0.037, 0.007, 0.0]
N: [6, 75, 185, 126, 50, 13, 2, 1]
bin: [0.0, 0.25, 0.55, 0.8, 1.1, 1.35, 1.65, 1.9]
""",
    "real_arcade_DWB_efficiency_max.yaml": """\
site: Arcade
state: Real
baseline: DWB
metric: Efficiency
mean: [0.996, 0.989, 0.985, 0.992, 0.985]
std: [0.004, 0.008, 0.01, 0.003, 0.0]
N: [5, 30, 18, 4, 1]
bin: [0.25, 0.55, 0.8, 1.1, 1.35]
""",
    "real_arcade_Teleop_efficiency_max.yaml": """\
site: Arcade
state: Real
baseline: Teleop
metric: Efficiency
mean: [0.96, 0.97, 0.968, 0.973, 0.981]
std: [0.145, 0.096, 0.041, 0.027, 0.012]
N: []
bin: [0.0, 0.25, 0.55, 0.8, 1.1]
"""
}

# Create a BytesIO stream to hold the zip file in memory
zip_buffer = io.BytesIO()

# Write each YAML file into the zip file
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename, content in yamls.items():
        zf.writestr(filename, content)

# Write the zip file to disk
with open("real_arcade_yamls.zip", "wb") as f:
    f.write(zip_buffer.getvalue())
