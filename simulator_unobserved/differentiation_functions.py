from .backend import np

# This file contains functions related to differentiation reactions.

rng = np.random.default_rng()
poisson = rng.poisson


def cd44_reaction(
    expression_prev,
    location,
    p_cd44_base,
    cd44_base_rate_multiplier,
    d_cd44,
    dt,
):
    """
    Calculate the CD44 reaction based on the previous expression array .

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_cd44_base: Base parameter for CD44.
    - cd44_base_rate_multiplier: Multiplier for base CD44 parameter.
    - d_cd44: Optional parameter for CD44 differentiation, default is 0.
    - cd44_max: Maximum value for CD44 expression.
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CD44 reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)

    expression_new = (
        expression_prev
        + poisson(
            p_cd44_base * cd44_base_rate_multiplier * dt,
            size=int(len(expression_prev)),
        )
        - poisson(d_cd44 * expression_prev * dt, size=int(len(expression_prev)))
    )
    expression_new = np.clip(expression_new, 0, None)
    return expression_new


def ccr7_reaction(
    expression_prev,
    location,
    p_ccr7,
    ccr7_base_rate_multiplier,
    d_ccr7,
    d_2_ccr7,
    cell_compartment_dwell_time,
    K_dwell_time_ccr7,
    n_hill_ccr7,
    dt,
):
    """
    Calculate the CCR7 reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_ccr7: Parameter for CCR7 production.
    - ccr7_base_rate_multiplier: Multiplier for base CCR7 parameter.
    - d_ccr7: Parameter for CCR7 degradation.
    - d_2_ccr7: Additional degradation parameter for CCR7 when in lymph node or spleen.
    - cell_compartment_dwell_time: Time spent in the current compartment.
    - K_dwell_time_ccr7: Half-saturation constant for dwell time effect on
    - n_hill_ccr7: Hill coefficient for dwell time effect on CCR7
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CCR7 reaction.
    """

    expression_new = np.zeros_like(expression_prev, dtype=np.int32)
    filter = (location == 1) | (location == 3)

    rate = poisson(
        p_ccr7 * ccr7_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
    )
    hill_factor = (cell_compartment_dwell_time[filter] ** n_hill_ccr7) / (
        K_dwell_time_ccr7**n_hill_ccr7
        + cell_compartment_dwell_time[filter] ** n_hill_ccr7
    )

    # print("Dwell time:", cell_compartment_dwell_time[filter])
    # print("Hill factor:", hill_factor)
    # print("location:", location[filter])
    rate -= poisson(
        (d_ccr7 + d_2_ccr7 * hill_factor) * expression_prev[filter] * dt,
        size=int(np.sum(filter)),
    )

    expression_new[filter] = expression_prev[filter] + rate

    filter = (location != 1) & (location != 3)
    expression_new[filter] = (
        expression_prev[filter]
        + poisson(
            p_ccr7 * ccr7_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
        )
        - poisson(d_ccr7 * expression_prev[filter] * dt, size=int(np.sum(filter)))
    )

    x = np.clip(expression_new, 0, None)
    return x


def s1pr1_reaction(
    expression_prev,
    location,
    p_s1pr1,
    s1pr1_base_rate_multiplier,
    d_s1pr1,
    d_2_s1pr1,
    expression_CD69,
    K_CD69_S1PR1,
    n_hill_CD69_S1PR1,
    cell_compartment_dwell_time,
    K_dwell_time_s1pr1,
    n_hill_s1pr1,
    dt,
):
    """
    Calculate the S1PR1 reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_s1pr1: Parameter for S1PR1 production.
    - s1pr1_base_rate_multiplier: Multiplier for base S1PR1 parameter.
    - d_s1pr1: Parameter for S1PR1 degradation.
    - d_2_s1pr1: Additional degradation parameter for S1PR1 when in blood.
    - expression_CD69: Current expression values of CD69 from expression_matrix, should be shape (n_cells,)
    - K_CD69_S1PR1: Half-saturation constant for CD69 effect on
    - n_hill_CD69_S1PR1: Hill coefficient for CD69 effect on S1PR1
    - cell_compartment_dwell_time: Time spent in the current compartment.
    - K_dwell_time_s1pr1: Half-saturation constant for dwell time effect
    - n_hill_s1pr1: Hill coefficient for dwell time effect on S1PR1
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after S1PR1 reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)
    expression_CD69 = expression_CD69.astype(np.int64)

    filter = location == 2
    hill_rate_dwell_time = (cell_compartment_dwell_time[filter] ** n_hill_s1pr1) / (
        K_dwell_time_s1pr1**n_hill_s1pr1
        + cell_compartment_dwell_time[filter] ** n_hill_s1pr1
    )
    # hill_rate_dwell_time = 1

    expression_new[filter] = (
        expression_prev[filter]
        + poisson(
            p_s1pr1 * s1pr1_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
        )
        - poisson(
            (d_s1pr1 + d_2_s1pr1 * hill_rate_dwell_time) * expression_prev[filter] * dt,
            size=int(np.sum(filter)),
        )
    )

    filter = location != 2
    hill_rate_CD69_S1PR1 = expression_CD69[filter] ** n_hill_CD69_S1PR1 / (
        K_CD69_S1PR1**n_hill_CD69_S1PR1 + expression_CD69[filter] ** n_hill_CD69_S1PR1
    )
    # hill_rate_CD69_S1PR1 = 0

    expression_new[filter] = (
        expression_prev[filter]
        + poisson(
            p_s1pr1 * s1pr1_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
        )
        - poisson(
            (d_s1pr1 + 1 * d_2_s1pr1 * hill_rate_CD69_S1PR1)
            * expression_prev[filter]
            * dt,
            size=int(np.sum(filter)),
        )
    )
    return np.clip(expression_new, 0, None)


def cxcr4_reaction(
    expression_prev,
    location,
    p_cxcr4,
    cxcr4_base_rate_multiplier,
    d_cxcr4,
    d_2_cxcr4,
    cell_compartment_dwell_time,
    K_dwell_time_cxcr4,
    n_hill_cxcr4,
    dt,
):
    """
    Calculate the CXCR4 reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_cxcr4: Parameter for CXCR4 production.
    - cxcr4_base_rate_multiplier: Multiplier for base CXCR4 parameter.
    - d_cxcr4: Parameter for CXCR4 degradation.
    - d_2_cxcr4: Additional degradation parameter for CXCR4 when in bone marrow.
    - cell_compartment_dwell_time: Time spent in the current compartment.
    - K_dwell_time_cxcr4: Half-saturation constant for dwell time effect
    - n_hill_cxcr4: Hill coefficient for dwell time effect on CXCR4
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CXCR4 reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)
    filter = location != 4

    expression_new[filter] = (
        expression_prev[filter]
        + poisson(
            p_cxcr4 * cxcr4_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
        )
        - poisson(d_cxcr4 * expression_prev[filter] * dt, size=int(np.sum(filter)))
    )

    filter = location == 4
    hill_rate_dwell_time = (cell_compartment_dwell_time[filter] ** n_hill_cxcr4) / (
        K_dwell_time_cxcr4**n_hill_cxcr4
        + cell_compartment_dwell_time[filter] ** n_hill_cxcr4
    )
    expression_new[filter] = (
        expression_prev[filter]
        + poisson(
            p_cxcr4 * cxcr4_base_rate_multiplier[filter] * dt, size=int(np.sum(filter))
        )
        - poisson(
            (d_cxcr4 + d_2_cxcr4 * hill_rate_dwell_time) * expression_prev[filter] * dt,
            size=int(np.sum(filter)),
        )
    )

    return np.clip(expression_new, 0, None)


def a4b7_reaction(
    expression_prev, location, p_a4b7, a4b7_base_rate_multiplier, d_a4b7, dt
):
    """
    Calculate the a4b7 reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_a4b7: Parameter for a4b7 production.
    - a4b7_base_rate_multiplier: Multiplier for base a4b7 parameter.
    - d_a4b7: Parameter for a4b7 degradation.
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after a4b7 reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)

    expression_new = (
        expression_prev
        + poisson(
            p_a4b7 * a4b7_base_rate_multiplier * dt, size=int(len(expression_prev))
        )
        - poisson(d_a4b7 * expression_prev * dt, size=int(len(expression_prev)))
    )

    return np.clip(expression_new, 0, None)


def cd69_reaction(expression_prev, location, p_cd69, p_cd69_multiplier, d_cd69, dt):
    """
    Calculate the CD69 reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_cd69: Parameter for CD69 production.
    - p_cd69_multiplier: Multiplier for base CD69 parameter.
    - d_cd69: Parameter for CD69 degradation.
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CD69 reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)

    expression_new = (
        expression_prev
        + poisson(p_cd69 * p_cd69_multiplier * dt, size=int(len(expression_prev)))
        - poisson(d_cd69 * expression_prev * dt, size=int(len(expression_prev)))
    )

    return np.clip(expression_new, 0, None)


def cd62l_reaction(expression_prev, location, p_cd62l, p_cd62l_multiplier, d_cd62l, dt):
    """
    Calculate the CD62L reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_cd62l: Parameter for CD62L production.
    - p_cd62l_multiplier: Multiplier for base CD62L parameter.
    - d_cd62l: Parameter for CD62L degradation.
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CD62L reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)

    expression_new = (
        expression_prev
        + poisson(p_cd62l * p_cd62l_multiplier * dt, size=int(len(expression_prev)))
        - poisson(d_cd62l * expression_prev * dt, size=int(len(expression_prev)))
    )

    return np.clip(expression_new, 0, None)


def Lambda_reaction(
    expression_prev, p_Lambda, location, p_Lambda_multiplier, d_Lambda, dt
):
    """
    Calculate the CD62L reaction based on the previous expression array and death status.

    Parameters:
    - expression_prev: Previous expression values from expression_matrix, should be shape (n_cells,)
    - location: current location of cells, should be shape (n_cells,).
    - p_cd62l: Parameter for CD62L production.
    - p_cd62l_multiplier: Multiplier for base CD62L parameter.
    - d_cd62l: Parameter for CD62L degradation.
    - expression_CD69: Current expression values of CD69 from expression_matrix, should be shape (n_cells,)
    - K_S1PR1_Lambda: Half-saturation constant for Lambda effect on
    - n_hill_Lambda_S1PR1: Hill coefficient for CD69 effect on S1PR1
    - dt: Time step for the reaction.

    Returns:
    - Updated expression array after CD62L reaction.
    """
    expression_new = np.zeros_like(expression_prev, dtype=np.int32)

    expression_new = (
        expression_prev
        + poisson(p_Lambda * p_Lambda_multiplier * dt, size=int(len(expression_prev)))
        - poisson(d_Lambda * expression_prev * dt, size=int(len(expression_prev)))
    )

    return np.clip(expression_new, 0, None)
