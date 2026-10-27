"""
For multiprocessing purposes this script expects a single integer argument after the script name, 
 which is used to select a subset of the parameter matrix to process. 
 The argument is expected to be an integer between 0 and 99, inclusive. 
Each integer corresponds to a batch of 100 parameter sets, 
so the total number of parameter sets processed by this script is 10,000 (100 batches of 100 parameter sets each).
"""


import sys
import os

# Add project root to path (for simulator module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trafficking_inference.CellVADIS import (
    groundtruth_check,
    Stitch_OT,
    Joint_OT,
    simulate_data,
)
from simulator import location_dict, marker_dict
import numpy as np


np.random.seed(40)
results_path = "trafficking_inference/method_testing/results/results_test/"




n_hills_transition = "n_hills_param"
K_S1PR1_xx_to_blood = "K_S1PR1_param"
n_hill_dwell_time = "n_hill_dwell_time_param"
params = {
    "n_cells": (10000, 100000),  # Number of cells in the simulation
    "dt": (0.01, 0.1),  # Time step (hours)
    "p_cd44_base": 5,  # Base CD44 expression rate
    "cd44_base_multiplier_sd": 0.4,  # SD for CD44 base rate multiplier
    "cd44_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in CD44 base rate multiplier
    "d_cd44_base": 0.1,  # CD44 degradation rate
    "p_cxcr4_base": 6,  # CXCR4 expression rate
    "cxcr4_base_multiplier_sd": 0.1,  # SD for CXCR4 base rate multiplier
    "cxcr4_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in CXCR4 base rate multiplier
    "d_cxcr4_base": 0.1,  # CXCR4 degradation rate
    "d_2_cxcr4": 0.5,  # Additional degradation rate for CXCR4 when in bone marrow
    "K_dwell_time_cxcr4": (
        2,
        16,
    ),  # Half-saturation constant for dwell time effect on CXCR4
    "n_hill_dwell_time_cxcr4": n_hill_dwell_time,  # Hill coefficient for dwell time effect on CXCR4
    "p_s1pr1_base": 12,  # S1PR1 expression rate
    "d_s1pr1_base": 0.1,  # S1PR1 degradation rate
    "s1pr1_base_multiplier_sd": 0.1,  # SD for S1PR1 base rate multiplier
    "d_2_s1pr1": 1,  # Additional degradation rate for S1PR1 when in blood
    "K_CD69_S1PR1": 60,  # Half-saturation constant for CD69 in the S1PR1 degradation
    "n_hill_cd69_s1pr1": 6,  # Hill coefficient for CD69 effect on S1PR1 degradation
    "K_dwell_time_s1pr1": (
        0.01,
        0.1,
    ),  # Half-saturation constant for dwell time effect on S1PR1
    "n_hill_dwell_time_s1pr1": n_hill_dwell_time,  # Hill coefficient for dwell time effect on S1PR1
    "p_ccr7_base": 8,  # CCR7 expression rate
    "ccr7_base_multiplier_sd": 0.1,  # SD for CCR7 base rate multiplier
    "ccr7_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in CCR7 base rate multiplier
    "d_ccr7_base": 0.1,  # CCR7 degradation rate
    "d_2_ccr7": 1,  # Additional degradation rate for CCR7 when in LN
    "K_dwell_time_ccr7": (
        2,
        16,
    ),  # Half-saturation constant for dwell time effect on CCR7
    "n_hill_dwell_time_ccr7": n_hill_dwell_time,  # Hill coefficient for dwell time effect on CCR7
    "p_cd69_base": 3,  # CD69 expression rate
    "d_cd69_base": 0.1,  # CD69 degradation rate
    "p_cd69_base_multiplier_sd": 0.5,  # SD for CD69 multiplier
    "cd69_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in CD69 base rate multiplier
    "p_cd62l_base": 8,  # CD62L expression rate
    "d_cd62l_base": 0.1,  # CD62L degradation rate
    "p_cd62l_base_multiplier_sd": 0.3,  # SD for CD62L multiplier
    "cd62l_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in CD62L base rate multiplier
    "p_a4b7_base": 4,  # a4b7 expression rate
    "d_a4b7_base": 0.1,  # a4b7 degradation rate
    "p_a4b7_base_multiplier_sd": 0.5,  # SD for a4b7 multiplier
    "a4b7_base_rate_multiplier_weight": (
        0.0,
        1,
    ),  # Weight for the higher mode in a4b7 base rate multiplier
    "blood_to_ln_rate": (1, 100),  # Rate of transition from blood to lymph nodes
    "K_CCR7_blood_to_ln": (
        40,
        160,
    ),  # Half-saturation constant for CCR7 in the transition
    "K_CD62L_blood_to_ln": (
        40,
        160,
    ),  # Half-saturation constant for CD62L in the transition
    "n_hill_CCR7_blood_to_ln": n_hills_transition,  # Hill coefficient for CCR7 in blood to LN transition
    "n_hill_CD62L_blood_to_ln": n_hills_transition,  # Hill coefficient for CD62L in blood to LN transition
    "ln_to_blood_rate": (0.01, 1),  # Rate of transition from lymph nodes to blood
    "K_S1PR1_ln_to_blood": K_S1PR1_xx_to_blood,  # Half-saturation constant for S1PR1 in the transition
    "K_CCR7_ln_to_blood": 70,  # Half-saturation constant for CCR7 in the transition
    "n_hill_S1PR1_ln_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in LN to blood transition
    "n_hill_CCR7_ln_to_blood": n_hills_transition,  # Hill coefficient for CCR7 in LN to blood transition
    "spleen_to_blood_rate": (0.01, 2),  # Rate of transition from spleen to blood
    "K_S1PR1_spleen_to_blood": K_S1PR1_xx_to_blood,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_spleen_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in spleen to blood transition
    "blood_to_spleen_rate": (1, 100),  # Rate of transition from blood to spleen
    "K_CCR7_blood_to_spleen": (
        40,
        160,
    ),  # Half-saturation constant for CCR7 in the transition
    "n_hill_CCR7_blood_to_spleen": n_hills_transition,  # Hill coefficient for CCR7 in blood to spleen transition
    "BM_to_blood_rate": (0.01, 1),  # Rate of transition from bone marrow to blood
    "K_S1PR1_BM_to_blood": K_S1PR1_xx_to_blood,  # Half-saturation constant for S1PR1 in the transition
    "K_CXCR4_BM_to_blood": (
        40,
        160,
    ),  # Half-saturation constant for CXCR4 in the transition
    "n_hill_S1PR1_BM_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in BM to blood transition
    "n_hill_CD44_BM_to_blood": n_hills_transition,  # Hill coefficient for CD44 in BM to blood transition
    "n_hill_CXCR4_BM_to_blood": n_hills_transition,  # Hill coefficient for CXCR4 in BM to blood transition
    "K_CD44_BM_to_blood": (
        40,
        160,
    ),  # Half-saturation constant for CD44 in the transition
    "blood_to_BM_rate": (1, 100),  # Rate of transition from blood to bone marrow
    "K_CXCR4_blood_to_BM": (
        40,
        160,
    ),  # Half-saturation constant for CXCR4 in the transition
    "n_hill_CXCR4_blood_to_BM": n_hills_transition,  # Hill coefficient for CXCR4 in blood to BM transition
    "K_CD44_blood_to_BM": (
        40,
        160,
    ),  # Half-saturation constant for CD44 in the transition
    "n_hill_CD44_blood_to_BM": n_hills_transition,  # Hill coefficient for CD44 in blood to BM transition
    "gut_to_blood_rate": (0.01, 1),  # Rate of transition from gut to blood
    "K_S1PR1_gut_to_blood": K_S1PR1_xx_to_blood,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_gut_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in gut to blood transition
    "blood_to_gut_rate": (1, 100),  # Rate of transition from blood to gut
    "K_a4b7_blood_to_gut": (
        40,
        160,
    ),  # Half-saturation constant for a4b7 in the transition
    "n_hill_a4b7_blood_to_gut": n_hills_transition,  # Hill coefficient for a4b7 in blood to gut transition
    "K_S1PR1_blood_to_xx": (
        20,
        80,
    ),  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_blood_to_xx": n_hills_transition,  # Hill coefficient for S1PR1 in blood to xx transition
}


# rad in parameter from python script call
#For multiprocessing purposes this script expects a single integer argument after the script name, 
# which is used to select a subset of the parameter matrix to process. 
# The argument is expected to be an integer between 0 and 99, inclusive. 
# Each integer corresponds to a batch of 100 parameter sets, 
# so the total number of parameter sets processed by this script is 10,000 (100 batches of 100 parameter sets each).
job_id = int(sys.argv[1])

print("Job ID:", job_id)
# Load parameter matrix
parameter_matrix = np.load(f"{results_path}parameter_matrix.npy")[
    job_id * 100 : (job_id + 1) * 100, :
]


# store results for each of the 4 methods
results = np.zeros(
    (parameter_matrix.shape[0], 12, 16)
)  # Simulation_id, method_id, statistics_id

real_fractions = np.zeros((parameter_matrix.shape[0], 4))


print(parameter_matrix.shape[0], "parameter sets to process.")
for i in range(parameter_matrix.shape[0]):
    print("Running parameter set ", i)
    params = dict(zip(params.keys(), parameter_matrix[i]))
    simulation_successful = False
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data_raw, data_normalized, locations, true_next_locations = simulate_data(
                params, procs=16
            )
            true_next_locations = true_next_locations[
                locations == 2
            ]  # only consider cells starting in blood
            for j, loc in enumerate([1, 3, 4, 5]):
                real_fractions[i, j] = np.sum(true_next_locations == loc) / len(
                    true_next_locations
                )

            data_source = data_normalized[
                locations == 2
            ]  # only consider cells starting in blood
            data_target = data_normalized[
                locations != 2
            ]  # only consider cells starting in non-blood locations
            target_locations = locations[
                locations != 2
            ]  # only consider cells starting in non-blood locations
            target_locations = np.array(
                [location_dict[loc] for loc in target_locations]
            )

            source_name = "Blood"
            target_names = ["LN", "Spleen", "BM", "Gut"]
            data_source_raw = data_raw[
                locations == 2
            ]  # only consider cells starting in blood
            fate_groups_ids = [1, 3, 4, 5]  # LN, Spleen, BM, Gut

            simulation_successful = True

            break  # Exit the retry loop if successful

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                print(
                    f"  All {max_retries} attempts failed for parameter set {i}. Skipping..."
                )
    if not simulation_successful:
        continue

    # check if any of the unique counts in true_next_locations is less than 100,
    # these simulations should be skipped since they are too sparse for the OT methods to work
    unique, counts = np.unique(locations, return_counts=True)
    print(
        f"Location counts for parameter set {i}: {np.unique(locations, return_counts=True)}"
    )
    if np.any(counts < 100):
        print(data_raw.shape)
        print(data_source.shape)
        print(true_next_locations.shape)
        print(np.unique(true_next_locations, return_counts=True))
        print(
            f"  Parameter set {i} has less than 100 cells in at least one of the next location compartments. Skipping..."
        )
        continue

    # save the data for this parameter set, all 4 arrays in a single npz file
    # create subfolder if it doesn't exist
    if not os.path.exists(f"{results_path}simulations/"):
        os.makedirs(f"{results_path}simulations/")
    np.savez(
        f"{results_path}simulations/simulation_data_{job_id}_{i}.npz",
        data_raw=data_raw,
        data_normalized=data_normalized,
        locations=locations,
        true_next_locations=true_next_locations,
    )

    # StitchOT tau_a=0.8
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.8,
        subsample_non_blood=True,
        rescale_OT_plans=True,
    )

    results[i, 0, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 0, ": ", results[i, 0, :])

    # StitchOT tau_a=0.95
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.95,
        subsample_non_blood=True,
        rescale_OT_plans=True,
    )

    results[i, 1, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 1, ": ", results[i, 1, :])

    # StitchOT tau_a=0.5
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.5,
        subsample_non_blood=True,
        rescale_OT_plans=True,
    )

    results[i, 2, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 2, ": ", results[i, 2, :])

    # StitchOT tau_a=0.8, no rescaling
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.8,
        subsample_non_blood=True,
        rescale_OT_plans=False,
    )

    results[i, 3, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 3, ": ", results[i, 3, :])

    # StitchOT tau_a=0.8, no subsampling
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.8,
        subsample_non_blood=False,
        rescale_OT_plans=True,
    )

    results[i, 4, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 4, ": ", results[i, 4, :])

    # only source and target cells with 33% of cells with lowest s1pr1
    filter_source = data_source[:, marker_dict["S1PR1"]] < np.percentile(
        data_source[:, marker_dict["S1PR1"]], 33
    )
    filter_target = data_target[:, marker_dict["S1PR1"]] < np.percentile(
        data_target[:, marker_dict["S1PR1"]], 33
    )
    data_source_low_s1pr1 = data_source[filter_source]
    data_target_low_s1pr1 = data_target[filter_target]
    target_locations_low_s1pr1 = target_locations[filter_target]
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source_low_s1pr1,
        data_target_low_s1pr1,
        target_locations_low_s1pr1,
        target_names,
        tau_a=0.8,
        subsample_non_blood=True,
        rescale_OT_plans=True,
    )

    results[i, 5, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw[filter_source],
        target_locations_low_s1pr1[selected_target_mask],
        target_names,
        params,
        true_next_locations[filter_source],
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 5, ": ", results[i, 5, :])

    # only source and target cells with 33% of cells with lowest s1pr1
    filter_target = data_target[:, marker_dict["S1PR1"]] < np.percentile(
        data_target[:, marker_dict["S1PR1"]], 33
    )
    data_source_low_s1pr1 = data_source
    data_target_low_s1pr1 = data_target[filter_target]
    target_locations_low_s1pr1 = target_locations[filter_target]
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source_low_s1pr1,
        data_target_low_s1pr1,
        target_locations_low_s1pr1,
        target_names,
        tau_a=0.8,
        subsample_non_blood=True,
        rescale_OT_plans=True,
    )

    results[i, 6, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations_low_s1pr1[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 6, ": ", results[i, 6, :])

    # Randomized OT for comparison to simple guessing
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Stitch_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_a=0.8,
        subsample_non_blood=True,
        rescale_OT_plans=True,
        random_shuffle=True,
    )

    results[i, 7, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 7, ": ", results[i, 7, :])

    # Joint OT tau_b=0.8
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Joint_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_b=0.8,
        subsample_non_blood=True,
        use_true_fractions=None,
    )

    results[i, 8, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 8, ": ", results[i, 8, :])

    # Joint OT tau_b=0.95
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Joint_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_b=0.95,
        subsample_non_blood=True,
        use_true_fractions=None,
    )

    results[i, 9, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 9, ": ", results[i, 9, :])

    # Joint OT tau_b=0.5
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Joint_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_b=0.5,
        subsample_non_blood=True,
        use_true_fractions=None,
    )

    results[i, 10, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 10, ": ", results[i, 10, :])

    # Joint OT tau_b=0.8, true fractions
    fate_matrix, coupling_matrix, OT_plan, selected_target_mask = Joint_OT(
        data_source,
        data_target,
        target_locations,
        target_names,
        tau_b=0.8,
        subsample_non_blood=True,
        use_true_fractions=real_fractions[i],
    )

    results[i, 11, :] = groundtruth_check(
        fate_matrix,
        OT_plan,
        data_source_raw,
        target_locations[selected_target_mask],
        target_names,
        params,
        true_next_locations,
        fate_groups_ids,
    )

    print(" Results for parameter set ", i, " method ", 11, ": ", results[i, 11, :])


# except Exception as e:
#     print(f"  Error processing parameter set {i}: {e}. Skipping...")
#     continue


np.save(f"{results_path}results_{job_id}.npy", results)
