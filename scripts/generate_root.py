import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
# from pathlib import Path
import uproot
import os
from hist import Hist

def filename_to_dirname(filename):
    # Strip 'run-' prefix, '.parquet' suffix, and remove underscore
    base = filename.removeprefix("run-").removesuffix(".parquet")
    run_parts = base.split("_")
    return "Run" + "".join(run_parts)

# 3D plotting function
def plot_array(array, plot_title, save_path=None):
    x, y, z = np.meshgrid(np.arange(array.shape[0]),
                          np.arange(array.shape[1]),
                          np.arange(array.shape[2]), indexing='ij')
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')
    cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0 - 0.04,
                        ax.get_position().width, 0.02])
    scatter = ax.scatter(y.flatten(), x.flatten(), z.flatten(), c=array.flatten(), marker=".", cmap='Spectral_r')
    fig.colorbar(scatter, orientation="horizontal", cax=cax, pad=1)
    ax.set_ylabel('LS')
    ax.set_xlabel('ieta')
    ax.set_zlabel('iphi')
    ax.set_title(plot_title)
    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight')
    #     print(f"Saved 3D plot: {save_path}")
    # plt.close()
    plt.show()

# Load the parquet file
def process_parquet_file(file_path, me_name):

    # Read the parquet file
    df = pq.read_table(file_path).to_pandas()
    
    filtered_df = df[df['me'] == me_name]
    
    # Get unique ls_number values
    unique_ls_numbers = filtered_df['ls_number'].unique()

    run_number = df["run_number"].iloc[0] if "run_number" in df.columns else 0
    
    # Create a dictionary to store data for each unique ls_number
    result_list = []

    # For each unique ls_number, extract the corresponding data
    for ls_num in unique_ls_numbers:

        # Get data for this ls_number
        data_for_ls = filtered_df[filtered_df['ls_number'] == ls_num]['data']
        data_np = np.stack(data_for_ls.values[0]).T
        data_np = data_np[10:-10, :]  # From (84, 72) → (64, 72)
        result_list.append(data_np)

    result_list = np.stack(result_list)

    return result_list, unique_ls_numbers, run_number

# file_path = "run-380_744.parquet"
file_paths = ["run-387_721.parquet",
            "run-378_239.parquet",
            "run-379_530.parquet",
            "run-380_092.parquet",
            "run-380_481.parquet",
            "run-380_744.parquet",
            "run-381_208.parquet",
            "run-381_698.parquet",
            "run-382_465.parquet",
            "run-382_770.parquet",
            "run-383_254.parquet",
            "run-383_687.parquet",
            "run-384_052.parquet",
            "run-384_331.parquet",
            "run-384_963.parquet",
            "run-385_286.parquet",
            "run-385_712.parquet",
            "run-386_143.parquet",
            "run-386_673.parquet",
            "run-386_951.parquet"]

# file_path = "run-378_239.parquet"

for file_path in file_paths:

    depths = {"Hcal/DigiTask/OccupancyCut/depth/depth1": "hist3D_depth1",
            "Hcal/DigiTask/OccupancyCut/depth/depth2": "hist3D_depth2",
            "Hcal/DigiTask/OccupancyCut/depth/depth3": "hist3D_depth3",
            "Hcal/DigiTask/OccupancyCut/depth/depth4": "hist3D_depth4",
            "Hcal/DigiTask/OccupancyCut/depth/depth5": "hist3D_depth5",
            "Hcal/DigiTask/OccupancyCut/depth/depth6": "hist3D_depth6",
            "Hcal/DigiTask/OccupancyCut/depth/depth7": "hist3D_depth7"}

    # depths = {"Hcal/DigiTask/OccupancyCut/depth/depth1": "hist3D_depth1"}

    unique_ls_numbers = None
    run_number = None
    result_list = {}

    for depth in depths:

        result, unique_ls_numbers, run_number = process_parquet_file(file_path, me_name=depth)
        print(f"Histogram shape: {result.shape}")
        result_list[depth] = result

    # file_stem = Path(file_path).stem
    output_dir = filename_to_dirname(file_path)

    print(f"Total number of LS: {len(unique_ls_numbers)}")
    # Create output directory structure
    os.makedirs(f"{output_dir}/OccupancyCut", exist_ok=True)

    with uproot.recreate(f"{output_dir}/OccupancyCut/output.root") as occ_file:

        # Create directories in ROOT files
        occ_dir = occ_file.mkdir("Hcal4DQMAnalyzer")

        # Prepare arrays to hold data for all lumisections
        # Format: (eta, phi, depth, ls)
        num_ls = len(unique_ls_numbers)
        eta_bins = 64  # After trimming
        phi_bins = 72
        depth_bins = 7

        # Store run number and lumisection information
        run_nums = np.full(num_ls, run_number, dtype=np.int32)
        lumi_secs = np.array(unique_ls_numbers, dtype=np.int32)

        # Arrays for all depths and lumisections
        occ_data = np.zeros((eta_bins, phi_bins, depth_bins, num_ls), dtype=np.int32)
        occ_cut_data = np.zeros((eta_bins, phi_bins, depth_bins, num_ls), dtype=np.int32)

        for depth in depths:

            data = result_list[depth]  # Shape: (num_ls, 64, 72)
            name = depths[depth]       # e.g. "depth1"

            # data = data.astype(np.int64)
            print(f"data type: {data.dtype}")
            h = (
                Hist.new
                .Reg(data.shape[0], 0, data.shape[0], name="ls")
                .Reg(64, -32, 32, name="ieta")
                .Reg(72, 0, 72, name="iphi")
                .Double()
            )

            h[...] = data

            # Save histogram into ROOT file
            occ_dir[f"{name}"] = h

        # Create TTrees for run and lumisection info
        occ_tree = occ_dir.mktree("evttree", {"RunNum": "int32", "LumiSec": "int32"})

        # Fill TreeBranches
        occ_tree.extend({"RunNum": run_nums, "LumiSec": lumi_secs})