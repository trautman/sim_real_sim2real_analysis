import pandas as pd
import yaml
import ast

# Path to the spreadsheet file
file_path = "arcade-perks-ideal_human-data-vectors_max-density.xlsx"

def try_parse_list(val):
    """
    If val is a string representing a list, convert it to a Python list.
    Otherwise, return val as is.
    """
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed
        except Exception:
            return val
    else:
        return val

def extract_metrics(sheet_df):
    """
    Extract metric lists from a sheet DataFrame (without header).
    Expected structure:
      Row 0: "Xbins" -> bins
      Row 1: "N" -> counts
      Row 2: "Path Efficiency (means)" -> path efficiency means
      Row 3: "Path Efficiency (SDs)" -> path efficiency std
      Row 4: "Mean Safety Margin (means)" -> mean safety margin means
      Row 5: "Mean Safety Margin (SDs)" -> mean safety margin std
      Row 6: "Min Safety Margin (means)" -> min safety margin means
      Row 7: "Min Safety Margin (SDs)" -> min safety margin std
      Row 8: "Speed (means)" -> speed means
      Row 9: "Speed (SDs)" -> speed std
    """
    bins = try_parse_list(sheet_df.iloc[0, 1])
    N = try_parse_list(sheet_df.iloc[1, 1])
    pe_mean = try_parse_list(sheet_df.iloc[2, 1])
    pe_std = try_parse_list(sheet_df.iloc[3, 1])
    safety_mean = try_parse_list(sheet_df.iloc[4, 1])
    safety_std = try_parse_list(sheet_df.iloc[5, 1])
    min_dist_mean = try_parse_list(sheet_df.iloc[6, 1])
    min_dist_std = try_parse_list(sheet_df.iloc[7, 1])
    speed_mean = try_parse_list(sheet_df.iloc[8, 1])
    speed_std = try_parse_list(sheet_df.iloc[9, 1])

    metrics = {}
    metrics['path_efficiency'] = {
        "site": None,  # to be set later
        "state": "Real",
        "baseline": "Human",
        "metric": "Path Efficiency",
        "mean": pe_mean,
        "std": pe_std,
        "N": N,
        "bin": bins
    }
    metrics['ave_safety_distance'] = {
        "site": None,
        "state": "Real",
        "baseline": "Human",
        "metric": "Ave Safety Distance",
        "mean": safety_mean,
        "std": safety_std,
        "N": N,
        "bin": bins
    }
    metrics['min_dist_to_person'] = {
        "site": None,
        "state": "Real",
        "baseline": "Human",
        "metric": "Min Dist to Person",
        "mean": min_dist_mean,
        "std": min_dist_std,
        "N": N,
        "bin": bins
    }
    metrics['ave_translational_velocity'] = {
        "site": None,
        "state": "Real",
        "baseline": "Human",
        "metric": "Ave Translational Velocity",
        "mean": speed_mean,
        "std": speed_std,
        "N": N,
        "bin": bins
    }
    return metrics

# Process the "arcade" sheet
df_arcade = pd.read_excel(file_path, sheet_name="arcade", header=None)
arcade_metrics = extract_metrics(df_arcade)
for key in arcade_metrics:
    arcade_metrics[key]["site"] = "Arcade"

# Process the "perks" sheet
df_perks = pd.read_excel(file_path, sheet_name="perks", header=None)
perks_metrics = extract_metrics(df_perks)
for key in perks_metrics:
    perks_metrics[key]["site"] = "Perks"

# Process the "ideal" sheet (assumed to correspond to "Wharf")
df_wharf = pd.read_excel(file_path, sheet_name="ideal", header=None)
wharf_metrics = extract_metrics(df_wharf)
for key in wharf_metrics:
    wharf_metrics[key]["site"] = "Wharf"

# Define file name parts for each metric.
# Filename conventions:
#  - For path efficiency, use "efficiency_max.yaml"
#  - For ave safety distance, use "ave_safety_dist_max.yaml"
#  - For min dist to person, use "min_dist_to_person_max.yaml"
#  - For ave translational velocity, use "velocity_max.yaml"
file_names = {
    "path_efficiency": "efficiency_max.yaml",
    "ave_safety_distance": "ave_safety_dist_max.yaml",
    "min_dist_to_person": "min_dist_to_person_max.yaml",
    "ave_translational_velocity": "velocity_max.yaml"
}

def write_yaml_files(metrics_dict, site_name):
    """
    Write YAML files for a given site.
    Filenames will be of the form:
      real_{site_lower}_human_{metric_filename_part}
    """
    for metric_key, data in metrics_dict.items():
        filename = f"real_{site_name.lower()}_human_{file_names[metric_key]}"
        with open(filename, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"Wrote {filename}")

# Write YAML files for each site
write_yaml_files(arcade_metrics, "Arcade")
write_yaml_files(perks_metrics, "Perks")
write_yaml_files(wharf_metrics, "Wharf")

print("YAML files for Arcade, Perks, and Wharf have been generated successfully.")
