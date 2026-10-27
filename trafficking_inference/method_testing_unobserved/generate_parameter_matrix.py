import sys
import os
import numpy as np

# Add project root to path (for simulator module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Add trafficking_inference to path (for helper_functions)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np.random.seed(4)
results_path = "trafficking_inference/method_testing_unobserved/results/results_test/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

n_hills_transition = "n_hills_param"
K_S1PR1_xx_to_blood = "K_S1PR1_param"
n_hill_dwell_time = "n_hill_dwell_time_param"
params = {
    "n_cells": (10000, 100000),  # Number of cells in the simulation
    "dt": 0.02,  # Time step (hours)
    "p_cd44_base": 5,  # Base CD44 expression rate
    "cd44_base_multiplier_sd": 0.4,  # SD for CD44 base rate multiplier
    "cd44_base_rate_multiplier_weight": (
        0.1,
        1,
    ),  # Weight for the higher mode in CD44 base rate multiplier
    "d_cd44_base": 0.1,  # CD44 degradation rate
    "p_cxcr4_base": 6,  # CXCR4 expression rate
    "cxcr4_base_multiplier_sd": 0.1,  # SD for CXCR4 base rate multiplier
    "cxcr4_base_rate_multiplier_weight": (
        0.1,
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
        0.1,
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
        0.1,
        1,
    ),  # Weight for the higher mode in CD69 base rate multiplier
    "p_cd62l_base": 8,  # CD62L expression rate
    "d_cd62l_base": 0.1,  # CD62L degradation rate
    "p_cd62l_base_multiplier_sd": 0.3,  # SD for CD62L multiplier
    "cd62l_base_rate_multiplier_weight": (
        0.1,
        1,
    ),  # Weight for the higher mode in CD62L base rate multiplier
    "p_a4b7_base": 4,  # a4b7 expression rate
    "d_a4b7_base": 0.1,  # a4b7 degradation rate
    "p_a4b7_base_multiplier_sd": 0.5,  # SD for a4b7 multiplier
    "a4b7_base_rate_multiplier_weight": (
        0.1,
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
    "p_Lambda_base": 4,  # Lambda expression rate
    "d_Lambda_base": 0.1,  # Lambda
    "K_Lambda_blood_to_U": 50,  # Half-saturation constant for Lambda in the transition
    "Lambda_base_rate_multiplier_weight": (
        0,
        1.0,
    ),  # Weight for the base rate multiplier in Lambda expression
    "K_Lambda_blood_to_xx": 50,  # Half-saturation constant for Lambda in the transition
    "Lambda_steepness": (0, 1.0),
    "blood_to_U_rate": 30 * 0.2,  # Rate of transition from blood to U
    "U_to_blood_rate": 0.2,  # Rate of transition from U to blood
    "K_S1PR1_U_to_blood": K_S1PR1_xx_to_blood,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_U_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in U to blood transition
}

n_parameter_sets = 1000  # Number of random parameter sets per grid point
grid_size = 10  # Number of values per axis for the systematic grid

parameter_keys = list(params.keys())
grid_Lambda_base_rate_multiplier = np.linspace(
    params["Lambda_base_rate_multiplier_weight"][0],
    params["Lambda_base_rate_multiplier_weight"][1],
    20,
)
grid_Lambda_steepness = np.logspace(
    -3,
    0,
    grid_size,
)

parameter_matrix = np.zeros((grid_size * 20 * n_parameter_sets, len(params)))

# Sample the non-grid parameters once, then reuse the exact same parameter sets
# for every grid point.
base_parameter_sets = []
for _ in range(n_parameter_sets):
    sampled_params = {}
    n_hills_transition = np.random.uniform(1, 6)
    n_hill_dwell_time = np.random.uniform(1, 6)
    K_S1PR1_xx_to_blood = np.random.uniform(40, 160)
    for key in parameter_keys:
        if isinstance(params[key], tuple):
            sampled_params[key] = np.random.uniform(params[key][0], params[key][1])
        elif isinstance(params[key], str) and params[key] == "n_hills_param":
            sampled_params[key] = n_hills_transition
        elif isinstance(params[key], str) and params[key] == "n_hill_dwell_time_param":
            sampled_params[key] = n_hill_dwell_time
        elif isinstance(params[key], str) and params[key] == "K_S1PR1_param":
            sampled_params[key] = K_S1PR1_xx_to_blood
        else:
            sampled_params[key] = params[key]

    base_parameter_sets.append(sampled_params)

row = 0
for Lambda_base_rate_multiplier in grid_Lambda_base_rate_multiplier:
    for Lambda_steepness in grid_Lambda_steepness:
        for sampled_params in base_parameter_sets:
            for col, key in enumerate(parameter_keys):
                if key == "Lambda_base_rate_multiplier_weight":
                    parameter_matrix[row, col] = Lambda_base_rate_multiplier
                elif key == "Lambda_steepness":
                    parameter_matrix[row, col] = Lambda_steepness
                else:
                    parameter_matrix[row, col] = sampled_params[key]
            row += 1

np.save(f"{results_path}parameter_matrix.npy", parameter_matrix)

# check for correctness
print("Parameter matrix shape:", parameter_matrix.shape)
print("Parameter matrix sample:", parameter_matrix[:20, -5])
print("Parameter matrix sample:", parameter_matrix[:20, -7])
