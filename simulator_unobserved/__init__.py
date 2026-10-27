import seaborn as sns

# model params
n_hills_transition = 3  # Hill coefficient for all trafficking transitions
params = {
    "p_cd44_base": 5,  # Base CD44 expression rate
    "cd44_base_multiplier_sd": 0.4,  # SD for CD44 base rate multiplier
    "cd44_base_rate_multiplier_weight": 0.5,  # Weight for the base rate multiplier in CD44 expression
    "d_cd44_base": 0.1,  # CD44 degradation rate
    "p_cxcr4_base": 6,  # CXCR4 expression rate
    "cxcr4_base_multiplier_sd": 0.1,  # SD for CXCR4 base rate multiplier
    "cxcr4_base_rate_multiplier_weight": 0.2,  # Weight for the base rate multiplier in CXCR4 expression
    "d_cxcr4_base": 0.1,  # CXCR4 degradation rate
    "d_2_cxcr4": 0.5,  # Additional degradation rate for CXCR4 when in bone marrow
    "K_dwell_time_cxcr4": 16,  # Half-saturation constant for dwell time effect on CXCR4
    "n_hill_dwell_time_cxcr4": 4,  # Hill coefficient for dwell time effect on CXCR4
    "p_s1pr1_base": 12,  # S1PR1 expression rate
    "d_s1pr1_base": 0.1,  # S1PR1 degradation rate
    "s1pr1_base_multiplier_sd": 0.1,  # SD for S1PR1 base rate multiplier
    "d_2_s1pr1": 1,  # Additional degradation rate for S1PR1 when in blood
    "K_CD69_S1PR1": 60,  # Half-saturation constant for CD69 in the S1PR1 degradation
    "n_hill_cd69_s1pr1": 6,  # Hill coefficient for CD69 effect on S1PR1 degradation
    "K_dwell_time_s1pr1": 0.05,  # Half-saturation constant for dwell time effect on S1PR1
    "n_hill_dwell_time_s1pr1": 4,  # Hill coefficient for dwell time effect on S1PR1
    "p_ccr7_base": 8,  # CCR7 expression rate
    "ccr7_base_multiplier_sd": 0.1,  # SD for CCR7 base rate multiplier
    "ccr7_base_rate_multiplier_weight": 0.3,  # Weight for the base rate multiplier in CCR7 expression
    "d_ccr7_base": 0.1,  # CCR7 degradation rate
    "d_2_ccr7": 1,  # Additional degradation rate for CCR7 when in LN
    "K_dwell_time_ccr7": 8,  # Half-saturation constant for dwell time effect on CCR7
    "n_hill_dwell_time_ccr7": 4,  # Hill coefficient for dwell time effect on CCR7
    "p_cd69_base": 3,  # CD69 expression rate
    "d_cd69_base": 0.1,  # CD69 degradation rate
    "p_cd69_base_multiplier_sd": 0.5,  # SD for CD69 multiplier
    "cd69_base_rate_multiplier_weight": 0.8,  # Weight for the base rate multiplier in CD69 expression
    "p_cd62l_base": 8,  # CD62L expression rate
    "d_cd62l_base": 0.1,  # CD62L degradation rate
    "p_cd62l_base_multiplier_sd": 0.3,  # SD for CD62L multiplier
    "cd62l_base_rate_multiplier_weight": 0.7,  # Weight for the base rate multiplier in CD62L expression
    "p_a4b7_base": 4,  # a4b7 expression rate
    "d_a4b7_base": 0.1,  # a4b7 degradation rate
    "p_a4b7_base_multiplier_sd": 0.5,  # SD for a4b7 multiplier
    "a4b7_base_rate_multiplier_weight": 0.5,  # Weight for the base rate multiplier in a4b7 expression
    "p_Lambda_base": 4,  # Lambda expression rate
    "d_Lambda_base": 0.1,  # Lambda
    "K_Lambda_blood_to_U": 60,  # Half-saturation constant for Lambda in the transition
    "Lambda_base_rate_multiplier_weight": 0.80,  # Weight for the base rate multiplier in Lambda expression
    "K_Lambda_blood_to_xx": 50,  # Half-saturation constant for Lambda in the transition
    "Lambda_steepness": 0.1,
    "blood_to_ln_rate": 30 * 1,  # Rate of transition from blood to lymph nodes
    "K_CCR7_blood_to_ln": 100,  # Half-saturation constant for CCR7 in the transition
    "K_CD62L_blood_to_ln": 130,  # Half-saturation constant for CD62L in the transition
    "n_hill_CCR7_blood_to_ln": n_hills_transition,  # Hill coefficient for CCR7 in blood to LN transition
    "n_hill_CD62L_blood_to_ln": n_hills_transition,  # Hill coefficient for CD62L in blood to LN transition
    "ln_to_blood_rate": 0.07,  # Rate of transition from lymph nodes to blood
    "K_S1PR1_ln_to_blood": 60,  # Half-saturation constant for S1PR1 in the transition
    "K_CCR7_ln_to_blood": 70,  # Half-saturation constant for CCR7 in the transition
    "n_hill_S1PR1_ln_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in LN to blood transition
    "n_hill_CCR7_ln_to_blood": n_hills_transition,  # Hill coefficient for CCR7 in LN to blood transition
    "spleen_to_blood_rate": 1.5,  # Rate of transition from spleen to blood
    "K_S1PR1_spleen_to_blood": 60,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_spleen_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in spleen to blood transition
    "blood_to_spleen_rate": 30 * 0.5,  # Rate of transition from blood to spleen
    "K_CCR7_blood_to_spleen": 100,  # Half-saturation constant for CCR7 in the transition
    "n_hill_CCR7_blood_to_spleen": n_hills_transition,  # Hill coefficient for CCR7 in blood to spleen transition
    "BM_to_blood_rate": 0.04,  # Rate of transition from bone marrow to blood
    "K_S1PR1_BM_to_blood": 60,  # Half-saturation constant for S1PR1 in the transition
    "K_CXCR4_BM_to_blood": 120,  # Half-saturation constant for CXCR4 in the transition
    "n_hill_S1PR1_BM_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in BM to blood transition
    "n_hill_CD44_BM_to_blood": n_hills_transition,  # Hill coefficient for CD44 in BM to blood transition
    "n_hill_CXCR4_BM_to_blood": n_hills_transition,  # Hill coefficient for CXCR4 in BM to blood transition
    "K_CD44_BM_to_blood": 100,  # Half-saturation constant for CD44 in the transition
    "blood_to_BM_rate": 30 * 0.3,  # Rate of transition from blood to bone marrow
    "K_CXCR4_blood_to_BM": 120,  # Half-saturation constant for CXCR4 in the transition
    "n_hill_CXCR4_blood_to_BM": n_hills_transition,  # Hill coefficient for CXCR4 in blood to BM transition
    "K_CD44_blood_to_BM": 90,  # Half-saturation constant for CD44 in the transition
    "n_hill_CD44_blood_to_BM": 2,  # Hill coefficient for CD44 in blood to BM transition
    "gut_to_blood_rate": 0.2,  # Rate of transition from gut to blood
    "K_S1PR1_gut_to_blood": 60,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_gut_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in gut to blood transition
    "blood_to_gut_rate": 30 * 0.2,  # Rate of transition from blood to gut
    "K_a4b7_blood_to_gut": 80,  # Half-saturation constant for a4b7 in the transition
    "n_hill_a4b7_blood_to_gut": n_hills_transition,  # Hill coefficient for a4b7 in blood to gut transition
    "K_S1PR1_blood_to_xx": 40,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_blood_to_xx": n_hills_transition,  # Hill coefficient for S1PR1 in blood to xx transition
    "blood_to_U_rate": 30 * 0.2,  # Rate of transition from blood to U
    "n_hill_Lambda_blood_to_U": 6,  # Hill coefficient for Lambda in blood to U transition
    "U_to_blood_rate": 0.2,  # Rate of transition from U to blood
    "K_S1PR1_blood_to_U": 40,  # Half-saturation constant for S1PR1 in the transition
    "K_S1PR1_U_to_blood": 40,  # Half-saturation constant for S1PR1 in the transition
    "n_hill_S1PR1_U_to_blood": n_hills_transition,  # Hill coefficient for S1PR1 in U to blood transition
}


marker_dict = {
    "CD44": 0,
    "Lambda": 1,
    "CCR7": 2,
    "S1PR1": 3,
    "CXCR4": 4,
    "CD69": 5,
    "CD62L": 6,
    "a4b7": 7,
    "Location": 8,
}

location_dict = {1: "LN", 2: "Blood", 3: "Spleen", 4: "BM", 5: "Gut", 6: "U"}

# get the 1,2,3,4 and 7th element of the colorblind palette
colors = sns.color_palette("colorblind")
# reorder such that 0,3,1,4,2
colors = ["#0072B2", "#D55E00", "#E69F00", "#CC79A7", "#009E73", "#B3B3B3"]
# create a dict mapping colors to locations
location_colors_dict = {i: colors[i - 1] for i in location_dict.keys()}
