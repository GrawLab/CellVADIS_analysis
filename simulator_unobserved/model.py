import numpy as np
from multiprocessing import Pool
# For the locations the compartments are coded as follows:
# 1: Lymph node
# 2: Blood
# 3: Spleen
# 4: Bone marrow
# 5: Gut


def simulate_model(
    params,
    t_end=24 * 7,
    n_cells=100,
    n_cells_max=1000,
    return_samples=None,
    dt=0.05,
    simulate_trafficking=True,
    use_gpu=False,
):
    """
    Simulates marker-driven T cell differentiation and migration.

    Parameters:
    - params: A dictionary containing parameter names and their values used for the simulation.
    - t_end: The end time for the simulation in hours. Will be overriden by max(return_samples) if return_samples is provided.
    - n_cells: The number of cells in the simulation.
    - n_cells_max: The maximum number of cells allowed in the simulation.
    - return_samples: If provided, the function will only store and return samples at specified time points.
                        If None, all time points will be stored.
    - dt: Time step for the simulation in hours.
    - simulate_trafficking: If False, migration between compartments is skipped (used for testing cell processes only).
    - use_gpu: If True, the simulation will utilize GPU acceleration via CuPy. Ensure that a compatible GPU is available.

    Returns:
    - results: A numpy array containing the simulation results.
    """
    from simulator_unobserved.backend import set_backend

    set_backend(use_gpu)
    from simulator_unobserved.backend import np
    import simulator_unobserved.differentiation_functions as diff_funcs
    import simulator_unobserved.migration_functions as mig_funcs

    t_transient = 24 * 2  # Transient time before reaching steady state
    if return_samples is not None:
        t_end = max(return_samples)  # Ensure we simulate until the last sample time
    t = np.arange(0, t_end + t_transient + dt, dt)
    n_steps = len(t)
    n_steps_transient = int(t_transient // dt)

    if return_samples is not None:
        expression_matrix = np.zeros(
            (len(return_samples) + 1, n_cells_max, 9),
            dtype=np.uint16,
        )
    else:
        expression_matrix = np.ones(
            (n_steps - n_steps_transient, n_cells_max, 9),
            dtype=np.uint16,
        )
    CD44_index = 0
    Lambda_index = 1
    CCR7_index = 2
    S1PR1_index = 3
    CXCR4_index = 4
    CD69_index = 5
    CD62L_index = 6
    a4b7_index = 7
    location_index = 8

    cell_compartment_dwell_time = np.zeros(
        np.shape(expression_matrix)[1], dtype=np.float32
    )
    base_rate_multipliers = np.ones(
        (9, np.shape(expression_matrix)[1]),
        dtype=np.float32,
    )

    def sample_bimodal(
        n_samples,
        weight=0.5,
        mean_lognormal=-1,
        sigma_lognormal=0.9,
        mean_normal=2,
        sigma_normal=0.5,
    ):
        n_lognormal = int(n_samples * weight)
        n_normal = n_samples - n_lognormal
        samples_lognormal = np.random.lognormal(
            mean=mean_lognormal, sigma=sigma_lognormal, size=n_lognormal
        )
        samples_normal = np.random.normal(
            loc=mean_normal, scale=sigma_normal, size=n_normal
        )
        # clip at 1
        samples_normal = np.clip(samples_normal, a_min=0.01, a_max=None)
        samples = np.concatenate([samples_lognormal, samples_normal])
        np.random.shuffle(samples)
        return samples

    # These arrays will be used to keep track of the cell states and age
    base_rate_multipliers[CD44_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["cd44_base_rate_multiplier_weight"],  # 0.70
        mean_lognormal=-1,
        sigma_lognormal=0.5,
        mean_normal=2,
        sigma_normal=0.3,
    )
    # store which cells have cd44 multiplier smaller than 1
    # these cells should have ccr7 and cd62l rates sampled from the normal distribution
    CD44_low_multiplier_cells = base_rate_multipliers[CD44_index, :n_cells] < 1
    base_rate_multipliers[CCR7_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["ccr7_base_rate_multiplier_weight"],  
        mean_lognormal=-3,
        sigma_lognormal=0.5,
        mean_normal=1.7,
        sigma_normal=0.3,
    )

    base_rate_multipliers[CCR7_index, :n_cells][CD44_low_multiplier_cells] = (
        np.random.normal(
            loc=1.7,
            scale=0.3,
            size=int(
                np.sum(CD44_low_multiplier_cells)
            ),  
        )
    )

    base_rate_multipliers[CCR7_index, :n_cells] = np.clip(
        base_rate_multipliers[CCR7_index, :n_cells], a_min=1e-05, a_max=None
    )

    base_rate_multipliers[S1PR1_index, :n_cells] = np.random.lognormal(
        mean=0, sigma=params["s1pr1_base_multiplier_sd"], size=n_cells
    )

    base_rate_multipliers[CXCR4_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["cxcr4_base_rate_multiplier_weight"],  # 0.20
        mean_lognormal=-1,
        sigma_lognormal=0.8,
        mean_normal=1.8,
        sigma_normal=0.8,
    )

    base_rate_multipliers[CD69_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["cd69_base_rate_multiplier_weight"],  # 0.80
        mean_lognormal=-1.5,
        sigma_lognormal=0.3,
        mean_normal=2.2,
        sigma_normal=0.5,
    )

    base_rate_multipliers[CD62L_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["cd62l_base_rate_multiplier_weight"],  # 0.30
        mean_lognormal=-1.5,
        sigma_lognormal=0.3,
        mean_normal=1.9,
        sigma_normal=0.4,
    )

    base_rate_multipliers[CD62L_index, :n_cells][CD44_low_multiplier_cells] = (
        np.random.normal(
            loc=1.9, scale=0.4, size=int(np.sum(CD44_low_multiplier_cells))
        )
    )

    # clip at 0.01 to avoid zeros
    base_rate_multipliers[CD62L_index, :n_cells] = np.clip(
        base_rate_multipliers[CD62L_index, :n_cells], a_min=1e-05, a_max=None
    )

    base_rate_multipliers[a4b7_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["a4b7_base_rate_multiplier_weight"],  # 0.50
        mean_lognormal=-1,
        sigma_lognormal=0.6,
        mean_normal=1.9,
        sigma_normal=0.4,
    )
    base_rate_multipliers[Lambda_index, :n_cells] = sample_bimodal(
        n_cells,
        weight=params["Lambda_base_rate_multiplier_weight"],  # 0.50
        mean_lognormal=-1,
        sigma_lognormal=0.6,
        mean_normal=2.1,
        sigma_normal=0.4,
    )

    expression_matrix[:, :n_cells, location_index] = 2  # All cells start in Blood

    n_current_cells = n_cells  # Current number of cells (alive + dead), to keep track of current matrix size
    current_index = 0
    for i in range(1, n_steps):
        j = i
        if i <= t_transient // dt:
            i = 1  # during transient period, do not fill matrix, just use the first two time steps
            expression_matrix[i - 1, :n_current_cells, :] = expression_matrix[
                i, :n_current_cells, :
            ]  # copy previous time step to current time step
        else:
            i = i - int(t_transient // dt)

            if return_samples is not None:
                if (
                    (i - 1) * dt in return_samples
                ):  # If previous time step was a return sample, store previous step in matrix
                    current_index = return_samples.index((i - 1) * dt) + 1

                    expression_matrix[
                        return_samples.index((i - 1) * dt),
                        :n_current_cells,
                        :,
                    ] = expression_matrix[
                        return_samples.index((i - 1) * dt) + 1, :n_current_cells, :
                    ]
                    if current_index == len(return_samples):
                        break
                else:
                    expression_matrix[
                        current_index,
                        :n_current_cells,
                        :,
                    ] = expression_matrix[current_index + 1, :n_current_cells, :]
                i = current_index + 1
        # restrict all matrices and arrays to only contain current generated cells
        # (so that we do not have to subset all by the current population size)
        current_expression_matrix = expression_matrix[:, :n_current_cells, :]
        current_cell_compartment_dwell_time = cell_compartment_dwell_time[
            :n_current_cells
        ]
        current_base_rate_multipliers = base_rate_multipliers[:, :n_current_cells]

        # MARKER DYNAMICS
        # CD44
        current_expression_matrix[i, :, CD44_index] = (
            diff_funcs.cd44_reaction( 
                current_expression_matrix[i - 1, :, CD44_index],
                current_expression_matrix[i - 1, :, location_index],
                params["p_cd44_base"],
                current_base_rate_multipliers[CD44_index, :],
                params["d_cd44_base"],
                dt,
            )
        )

        # Lambda
        current_expression_matrix[i, :, Lambda_index] = diff_funcs.Lambda_reaction(
            current_expression_matrix[i - 1, :, Lambda_index],
            params["p_Lambda_base"], 
            current_expression_matrix[i - 1, :, location_index],  
            current_base_rate_multipliers[Lambda_index, :], 
            params["d_Lambda_base"],  
            dt,  # dt
        )

        # S1PR1
        current_expression_matrix[i, :, S1PR1_index] = diff_funcs.s1pr1_reaction(
            current_expression_matrix[i - 1, :, S1PR1_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_s1pr1_base"],
            current_base_rate_multipliers[S1PR1_index, :],
            params["d_s1pr1_base"],
            params["d_2_s1pr1"],
            current_expression_matrix[i - 1, :, CD69_index],
            params["K_CD69_S1PR1"],
            params["n_hill_cd69_s1pr1"],
            current_cell_compartment_dwell_time,
            params["K_dwell_time_s1pr1"],
            params["n_hill_dwell_time_s1pr1"],
            dt,
        )
        # CCR7
        current_expression_matrix[i, :, CCR7_index] = diff_funcs.ccr7_reaction(
            current_expression_matrix[i - 1, :, CCR7_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_ccr7_base"],
            current_base_rate_multipliers[CCR7_index, :],
            params["d_ccr7_base"],
            params["d_2_ccr7"],
            current_cell_compartment_dwell_time,
            params["K_dwell_time_ccr7"],
            params["n_hill_dwell_time_ccr7"],
            dt,
        )
        # CXCR4
        current_expression_matrix[i, :, CXCR4_index] = diff_funcs.cxcr4_reaction(
            current_expression_matrix[i - 1, :, CXCR4_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_cxcr4_base"],
            current_base_rate_multipliers[CXCR4_index, :n_current_cells],
            params["d_cxcr4_base"],
            params["d_2_cxcr4"],
            current_cell_compartment_dwell_time,
            params["K_dwell_time_cxcr4"],
            params["n_hill_dwell_time_cxcr4"],
            dt,
        )
        # CD62L
        current_expression_matrix[i, :, CD62L_index] = diff_funcs.cd62l_reaction(
            current_expression_matrix[i - 1, :, CD62L_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_cd62l_base"],
            current_base_rate_multipliers[CD62L_index, :n_current_cells],
            params["d_cd62l_base"],
            dt,
        )
        # a4b7
        current_expression_matrix[i, :, a4b7_index] = diff_funcs.a4b7_reaction(
            current_expression_matrix[i - 1, :, a4b7_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_a4b7_base"],
            current_base_rate_multipliers[a4b7_index, :n_current_cells],
            params["d_a4b7_base"],
            dt,
        )

        # CD69
        current_expression_matrix[i, :, CD69_index] = diff_funcs.cd69_reaction(
            current_expression_matrix[i - 1, :, CD69_index],
            current_expression_matrix[i - 1, :, location_index],
            params["p_cd69_base"],
            current_base_rate_multipliers[CD69_index, :n_current_cells],
            params["d_cd69_base"],
            dt,
        )

        if simulate_trafficking:
            # MIGRATION DYNAMICS
            old_locations = current_expression_matrix[i - 1, :, location_index]
            new_locations = 2 * np.ones_like(old_locations)
            migrated = np.zeros_like(old_locations, dtype=bool)

            # define migration rules in a list
            rules_blood_to = [
                (
                    "blood_to_ln",
                    mig_funcs.blood_to_ln_transition,
                    {
                        "location_prev": old_locations,
                        "CCR7_expression": current_expression_matrix[
                            i - 1, :, CCR7_index
                        ],
                        "CD62L_expression": current_expression_matrix[
                            i - 1, :, CD62L_index
                        ],
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "Lambda_expression": current_expression_matrix[
                            i - 1, :, Lambda_index
                        ],
                        "blood_to_ln_rate": params["blood_to_ln_rate"],
                        "K_CCR7_blood_to_ln": params["K_CCR7_blood_to_ln"],
                        "K_CD62L_blood_to_ln": params["K_CD62L_blood_to_ln"],
                        "K_S1PR1_blood_to_ln": params["K_S1PR1_blood_to_xx"],
                        "n_hill_CCR7_blood_to_ln": params["n_hill_CCR7_blood_to_ln"],
                        "n_hill_CD62L_blood_to_ln": params["n_hill_CD62L_blood_to_ln"],
                        "n_hill_S1PR1_blood_to_ln": params["n_hill_S1PR1_blood_to_xx"],
                        "K_Lambda_blood_to_ln": params["K_Lambda_blood_to_xx"],
                        "steepness": params["Lambda_steepness"],
                        "dt": dt,
                    },
                ),
                (
                    "blood_to_spleen",
                    mig_funcs.blood_to_spleen_transition,
                    {
                        "location_prev": old_locations,
                        "CCR7_expression": current_expression_matrix[
                            i - 1, :, CCR7_index
                        ],
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "Lambda_expression": current_expression_matrix[
                            i - 1, :, Lambda_index
                        ],
                        "blood_to_spleen_rate": params["blood_to_spleen_rate"],
                        "K_CCR7_blood_to_spleen": params["K_CCR7_blood_to_spleen"],
                        "K_S1PR1_blood_to_spleen": params["K_S1PR1_blood_to_xx"],
                        "K_Lambda_blood_to_spleen": params["K_Lambda_blood_to_xx"],
                        "n_hill_CCR7_blood_to_spleen": params[
                            "n_hill_CCR7_blood_to_spleen"
                        ],
                        "n_hill_S1PR1_blood_to_spleen": params[
                            "n_hill_S1PR1_blood_to_xx"
                        ],
                        "steepness": params["Lambda_steepness"],
                        "dt": dt,
                    },
                ),
                (
                    "blood_to_BM",
                    mig_funcs.blood_to_BM_transition,
                    {
                        "location_prev": old_locations,
                        "CXCR4_expression": current_expression_matrix[
                            i - 1, :, CXCR4_index
                        ],
                        "CD44_expression": current_expression_matrix[
                            i - 1, :, CD44_index
                        ],
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "Lambda_expression": current_expression_matrix[
                            i - 1, :, Lambda_index
                        ],
                        "blood_to_BM_rate": params["blood_to_BM_rate"],
                        "K_CXCR4_blood_to_BM": params["K_CXCR4_blood_to_BM"],
                        "K_CD44_blood_to_BM": params["K_CD44_blood_to_BM"],
                        "K_S1PR1_blood_to_BM": params["K_S1PR1_blood_to_xx"],
                        "n_hill_CXCR4_blood_to_BM": params["n_hill_CXCR4_blood_to_BM"],
                        "n_hill_CD44_blood_to_BM": params["n_hill_CD44_blood_to_BM"],
                        "n_hill_S1PR1_blood_to_BM": params["n_hill_S1PR1_blood_to_xx"],
                        "K_Lambda_blood_to_BM": params["K_Lambda_blood_to_xx"],
                        "steepness": params["Lambda_steepness"],
                        "dt": dt,
                    },
                ),
                (
                    "blood_to_gut",
                    mig_funcs.blood_to_gut_transition,
                    {
                        "location_prev": old_locations,
                        "a4b7_expression": current_expression_matrix[
                            i - 1, :, a4b7_index
                        ],
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "Lambda_expression": current_expression_matrix[
                            i - 1, :, Lambda_index
                        ],
                        "blood_to_gut_rate": params["blood_to_gut_rate"],
                        "K_a4b7_blood_to_gut": params["K_a4b7_blood_to_gut"],
                        "K_S1PR1_blood_to_gut": params["K_S1PR1_blood_to_xx"],
                        "n_hill_a4b7_blood_to_gut": params["n_hill_a4b7_blood_to_gut"],
                        "n_hill_S1PR1_blood_to_gut": params["n_hill_S1PR1_blood_to_xx"],
                        "K_Lambda_blood_to_gut": params["K_Lambda_blood_to_xx"],
                        "steepness": params["Lambda_steepness"],
                        "dt": dt,
                    },
                ),
                (
                    "blood_to_U",
                    mig_funcs.blood_to_U_transition,
                    {
                        "location_prev": old_locations,
                        "Lambda_expression": current_expression_matrix[
                            i - 1, :, Lambda_index
                        ],
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "blood_to_U_rate": params["blood_to_U_rate"],
                        "K_Lambda_blood_to_U": params["K_Lambda_blood_to_xx"],
                        "steepness": params["Lambda_steepness"],
                        "n_hill_S1PR1_blood_to_U": params["n_hill_S1PR1_blood_to_xx"],
                        "K_S1PR1_blood_to_U": params["K_S1PR1_blood_to_xx"],
                        "dt": dt,
                    },
                ),
            ]

            rule_to_blood = [
                (
                    "ln_to_blood",
                    mig_funcs.ln_to_blood_transition,
                    {
                        "location_prev": old_locations,
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "CCR7_expression": current_expression_matrix[
                            i - 1, :, CCR7_index
                        ],
                        "ln_to_blood_rate": params["ln_to_blood_rate"],
                        "K_S1PR1_ln_to_blood": params["K_S1PR1_ln_to_blood"],
                        "K_CCR7_ln_to_blood": params["K_CCR7_ln_to_blood"],
                        "n_hill_S1PR1_ln_to_blood": params["n_hill_S1PR1_ln_to_blood"],
                        "n_hill_CCR7_ln_to_blood": params["n_hill_CCR7_ln_to_blood"],
                        "dt": dt,
                    },
                ),
                (
                    "U_to_blood",
                    mig_funcs.U_to_blood_transition,
                    {
                        "location_prev": old_locations,
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "U_to_blood_rate": params["U_to_blood_rate"],
                        "K_S1PR1_U_to_blood": params["K_S1PR1_U_to_blood"],
                        "n_hill_S1PR1_U_to_blood": params["n_hill_S1PR1_U_to_blood"],
                        "dt": dt,
                    },
                ),
                (
                    "spleen_to_blood",
                    mig_funcs.spleen_to_blood_transition,
                    {
                        "location_prev": old_locations,
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "spleen_to_blood_rate": params["spleen_to_blood_rate"],
                        "K_S1PR1_spleen_to_blood": params["K_S1PR1_spleen_to_blood"],
                        "n_hill_S1PR1_spleen_to_blood": params[
                            "n_hill_S1PR1_spleen_to_blood"
                        ],
                        "dt": dt,
                    },
                ),
                (
                    "BM_to_blood",
                    mig_funcs.BM_to_blood_transition,
                    {
                        "location_prev": old_locations,
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "CD44_expression": current_expression_matrix[
                            i - 1, :, CD44_index
                        ],
                        "CXCR4_expression": current_expression_matrix[
                            i - 1, :, CXCR4_index
                        ],
                        "BM_to_blood_rate": params["BM_to_blood_rate"],
                        "K_S1PR1_BM_to_blood": params["K_S1PR1_BM_to_blood"],
                        "K_CD44_BM_to_blood": params["K_CD44_BM_to_blood"],
                        "K_CXCR4_BM_to_blood": params["K_CXCR4_BM_to_blood"],
                        "n_hill_S1PR1_BM_to_blood": params["n_hill_S1PR1_BM_to_blood"],
                        "n_hill_CD44_BM_to_blood": params["n_hill_CD44_BM_to_blood"],
                        "n_hill_CXCR4_BM_to_blood": params["n_hill_CXCR4_BM_to_blood"],
                        "dt": dt,
                    },
                ),
                (
                    "gut_to_blood",
                    mig_funcs.gut_to_blood_transition,
                    {
                        "location_prev": old_locations,
                        "S1PR1_expression": current_expression_matrix[
                            i - 1, :, S1PR1_index
                        ],
                        "gut_to_blood_rate": params["gut_to_blood_rate"],
                        "K_S1PR1_gut_to_blood": params["K_S1PR1_gut_to_blood"],
                        "n_hill_S1PR1_gut_to_blood": params[
                            "n_hill_S1PR1_gut_to_blood"
                        ],
                        "dt": dt,
                    },
                ),
            ]
            transition_rates_blood_to = np.zeros(
                (
                    int(len(rules_blood_to)),
                    int(np.sum(old_locations == 2)),
                )
            )
            for j, (name, func, kwargs) in enumerate(
                rules_blood_to
            ):  # geht durch regel Liste durch
                rates = func(**kwargs)
                transition_rates_blood_to[j, :] = rates

            transition_rate_ln_to_blood = np.zeros(int(np.sum(old_locations == 1)))
            transition_rate_spleen_to_blood = np.zeros(int(np.sum(old_locations == 3)))
            transition_rate_BM_to_blood = np.zeros(int(np.sum(old_locations == 4)))
            transition_rate_gut_to_blood = np.zeros(int(np.sum(old_locations == 5)))
            transition_rate_U_to_blood = np.zeros(int(np.sum(old_locations == 6)))

            for j, (name, func, kwargs) in enumerate(rule_to_blood):
                rates = func(**kwargs)
                if rates.size == 0:
                    continue
                if name == "ln_to_blood":
                    transition_rate_ln_to_blood = rates
                    new_locations[old_locations == 1] = np.where(
                        np.random.binomial(1, 1 - np.exp(-transition_rate_ln_to_blood)),
                        2,
                        old_locations[old_locations == 1],
                    )
                elif name == "spleen_to_blood":
                    transition_rate_spleen_to_blood = rates
                    new_locations[old_locations == 3] = np.where(
                        np.random.binomial(
                            1, 1 - np.exp(-transition_rate_spleen_to_blood)
                        ),
                        2,
                        old_locations[old_locations == 3],
                    )
                elif name == "BM_to_blood":
                    transition_rate_BM_to_blood = rates
                    new_locations[old_locations == 4] = np.where(
                        np.random.binomial(1, 1 - np.exp(-transition_rate_BM_to_blood)),
                        2,
                        old_locations[old_locations == 4],
                    )
                elif name == "gut_to_blood":
                    transition_rate_gut_to_blood = rates
                    new_locations[old_locations == 5] = np.where(
                        np.random.binomial(
                            1, 1 - np.exp(-transition_rate_gut_to_blood)
                        ),
                        2,
                        old_locations[old_locations == 5],
                    )
                elif name == "U_to_blood":
                    transition_rate_U_to_blood = rates
                    new_locations[old_locations == 6] = np.where(
                        np.random.binomial(1, 1 - np.exp(-transition_rate_U_to_blood)),
                        2,
                        old_locations[old_locations == 6],
                    )
            sum_transition_rates_blood_to = np.sum(transition_rates_blood_to, axis=0)
            blood_stay_prob = np.exp(-sum_transition_rates_blood_to)

            sum_with_epsilon = np.where(
                sum_transition_rates_blood_to == 0,
                1e-64,
                sum_transition_rates_blood_to,
            )
            blood_to_ln_prob = (
                transition_rates_blood_to[0, :]
                / sum_with_epsilon
                * (1 - blood_stay_prob)
            )
            blood_to_spleen_prob = (
                transition_rates_blood_to[1, :]
                / sum_with_epsilon
                * (1 - blood_stay_prob)
            )
            blood_to_BM_prob = (
                transition_rates_blood_to[2, :]
                / sum_with_epsilon
                * (1 - blood_stay_prob)
            )
            blood_to_gut_prob = (
                transition_rates_blood_to[3, :]
                / sum_with_epsilon
                * (1 - blood_stay_prob)
            )

            blood_to_U_prob = (
                transition_rates_blood_to[4, :]
                / sum_with_epsilon
                * (1 - blood_stay_prob)
            )

            blood_indices = np.where(old_locations == 2)[0]
            rand_vals = np.random.uniform(0, 1, size=blood_indices.shape[0])

            thr1 = blood_to_ln_prob
            thr2 = thr1 + blood_to_spleen_prob
            thr3 = thr2 + blood_to_BM_prob
            thr4 = thr3 + blood_to_gut_prob
            thr5 = thr4 + blood_to_U_prob  # <= 1 - blood_stay_prob

            # boolean masks in the blood-sublocation space
            mask_ln = rand_vals < thr1
            mask_spleen = (rand_vals >= thr1) & (rand_vals < thr2)
            mask_BM = (rand_vals >= thr2) & (rand_vals < thr3)
            mask_gut = (rand_vals >= thr3) & (rand_vals < thr4)
            mask_U = (rand_vals >= thr4) & (rand_vals < thr5)

            # map masks back to global indices and assign new locations
            if mask_ln.any():
                idxs = blood_indices[mask_ln]
                new_locations[idxs] = 1
                migrated[idxs] = True
            if mask_spleen.any():
                idxs = blood_indices[mask_spleen]
                new_locations[idxs] = 3
                migrated[idxs] = True
            if mask_BM.any():
                idxs = blood_indices[mask_BM]
                new_locations[idxs] = 4
                migrated[idxs] = True
            if mask_gut.any():
                idxs = blood_indices[mask_gut]
                new_locations[idxs] = 5
                migrated[idxs] = True
            if mask_U.any():
                idxs = blood_indices[mask_U]
                new_locations[idxs] = 6
                migrated[idxs] = True

            current_cell_compartment_dwell_time[~migrated] += dt
            current_cell_compartment_dwell_time[migrated] = 0
            # commit locations
            current_expression_matrix[i, :, location_index] = new_locations

    # return as numpy arrays
    if use_gpu:
        expression_matrix = np.asnumpy(expression_matrix)
    return expression_matrix


def multiprocess_simulate_model(
    params,
    n_workers=4,
    t_end=24 * 7,
    n_cells=100,
    n_cells_max=10000,
    return_samples=None,
    dt=0.05,
    simulate_trafficking=True,
    use_gpu=False,
):
    n_cells_per_worker = n_cells // n_workers
    pool = Pool(processes=n_workers)
    results = []
    for i in range(n_workers):
        result = pool.apply_async(
            simulate_model,
            args=(
                params,
                t_end,
                n_cells_per_worker,
                n_cells_max // n_workers,
                return_samples,
                dt,
                simulate_trafficking,
                use_gpu,
            ),
        )
        results.append(result)
    pool.close()
    pool.join()
    # Combine results from all workers
    expression_matrices = [result.get() for result in results]
    combined_expression_matrix = np.concatenate(expression_matrices, axis=1)
    return combined_expression_matrix
