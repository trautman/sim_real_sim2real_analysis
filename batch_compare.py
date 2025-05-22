#!/usr/bin/env python3
import os, sys, glob, subprocess


def usage():
    print("Usage: batch_compare.py --label ped|robot <input_yaml_dir> <output_fig_dir>", file=sys.stderr)
    sys.exit(1)


def main():
    # 1) parse --label flag
    if len(sys.argv) < 2:
        usage()

    label = None
    args  = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--label":
            if i+1 >= len(args):
                usage()
            label = args[i+1]
            if label not in ("ped","robot"):
                print(f"Error: invalid label '{label}'", file=sys.stderr)
                usage()
            i += 2
        else:
            break

    # label is mandatory
    if not label:
        usage()

    # remaining positional args: input_dir and output_dir
    if len(args) - i != 2:
        usage()
    in_dir  = os.path.abspath(args[i])
    out_dir = os.path.abspath(args[i+1])
    os.makedirs(out_dir, exist_ok=True)

    # 2) choose the right compare script based on label
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if label == "ped":
        script = os.path.join(base_dir, "compare_pedestrian_sim2real.py")
    else:
        script = os.path.join(base_dir, "compare_robot_sim2real.py")
    if not os.path.isfile(script):
        print(f"Error: cannot find compare script at {script}", file=sys.stderr)
        sys.exit(1)

    # 3) collect all real_*.yaml and sim_*.yaml
    real_files = glob.glob(os.path.join(in_dir, "real_*.yaml"))
    sim_files  = glob.glob(os.path.join(in_dir, "sim_*.yaml"))

    # 4) index real files by (site, robot, metric)
    real_map = {}
    for fn in real_files:
        base = os.path.basename(fn)[:-5]
        parts = base.split("_", 3)
        if len(parts) == 4:
            _, site, robot, metric = parts
            real_map[(site, robot, metric)] = fn

    # 5) index sim files by same key
    sim_map = {}
    for fn in sim_files:
        base = os.path.basename(fn)[:-5]
        parts = base.split("_", 4)
        if len(parts) == 5:
            _, site, robot, simtype, metric = parts
            sim_map.setdefault((site, robot, metric), []).append(fn)

    # 6) loop through each group
    for key, real_fn in real_map.items():
        sims = sorted(sim_map.get(key, []))
        if len(sims) < 3:
            print(f"Skipping {key!r}: only {len(sims)} sim files")
            continue
        sims = sims[:3]

        # print summary
        print(f"▶ running compare for {key}")
        print(f"    real: {real_fn}")
        for idx, sim_fp in enumerate(sims, start=1):
            print(f"    sim{idx}: {sim_fp}")

        # build command
        cmd = [
            sys.executable,
            script,
            "--no-display",
            "--label", label,
            real_fn, sims[0], sims[1], sims[2]
        ]
        # run in output dir so figures land there
        subprocess.run(cmd, cwd=out_dir, check=True)


if __name__ == "__main__":
    main()
