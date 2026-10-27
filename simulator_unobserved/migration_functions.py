from .backend import np

# Cupy allows for much faster random number generation on GPU with large cell numbers (>10k)


# This file contains functions related to migration reactions.
def blood_to_U_transition(
    location_prev,
    Lambda_expression,
    S1PR1_expression,
    blood_to_U_rate,
    K_Lambda_blood_to_U,
    steepness,
    n_hill_S1PR1_blood_to_U,
    K_S1PR1_blood_to_U,
    dt,
):
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    Lambda_expression = Lambda_expression.astype(np.int64)

    transition_rate = (
        blood_to_U_rate
        * 1
        / (
            1
            + np.exp(
                -steepness
                * (Lambda_expression[location_prev == 2] - K_Lambda_blood_to_U)
            )
        )
        * K_S1PR1_blood_to_U**n_hill_S1PR1_blood_to_U
        / (
            K_S1PR1_blood_to_U**n_hill_S1PR1_blood_to_U
            + S1PR1_expression[location_prev == 2] ** n_hill_S1PR1_blood_to_U
        )
        * dt
    )

    return transition_rate


def U_to_blood_transition(
    location_prev,
    S1PR1_expression,
    K_S1PR1_U_to_blood,
    n_hill_S1PR1_U_to_blood,
    U_to_blood_rate,
    dt,
):
    """
    Calculate the transition of cells from lymph nodes to blood based on S1PR1 expression.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - S1PR1_expression: Expression values for S1PR1, should be shape (n_cells,).
    - ln_to_blood_rate: Rate of transition from lymph nodes to blood.
    - K_S1PR1_ln_to_blood: Half-saturation constant for S1PR1 in the transition.
    - n_hill_S1PR1_ln_to_blood: Hill coefficient for S1PR1 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    S1PR1_expression = S1PR1_expression.astype(np.int64)

    transition_rate = (
        U_to_blood_rate
        * S1PR1_expression[location_prev == 6] ** n_hill_S1PR1_U_to_blood
        / (
            K_S1PR1_U_to_blood**n_hill_S1PR1_U_to_blood
            + S1PR1_expression[location_prev == 6] ** n_hill_S1PR1_U_to_blood
        )
        * dt
    )
    return transition_rate


def blood_to_ln_transition(
    location_prev,
    CCR7_expression,
    CD62L_expression,
    S1PR1_expression,
    Lambda_expression,
    blood_to_ln_rate,
    K_CCR7_blood_to_ln,
    K_CD62L_blood_to_ln,
    K_Lambda_blood_to_ln,
    K_S1PR1_blood_to_ln,
    n_hill_CCR7_blood_to_ln,
    n_hill_CD62L_blood_to_ln,
    n_hill_S1PR1_blood_to_ln,
    steepness,
    dt,
):
    """
    Calculate the transition of cells from blood to lymph nodes based on CCR7 and S1PR1 expression.
    Later also CD62L, but behaviour of this marker mainly changes upon activation
    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - CCR7_expression: Expression values for CCR7, should be shape (n_cells,).
    - CD62L_expression: Expression values for CD62L, should be shape (n_cells,).
    - blood_to_ln_rate: Rate of transition from blood to lymph nodes.
    - K_CCR7_blood_to_ln: Half-saturation constant for CCR7 in the transition.
    - K_CD62L_blood_to_ln: Half-saturation constant for CD62L in the transition.
    - n_hill_CCR7_blood_to_ln: Hill coefficient for CCR7 in the transition.
    - n_hill_CD62L_blood_to_ln: Hill coefficient for CD62L in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    CCR7_expression = CCR7_expression.astype(np.int64)
    CD62L_expression = CD62L_expression.astype(np.int64)
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    Lambda_expression = Lambda_expression.astype(np.int64)

    transition_rate = (
        blood_to_ln_rate
        * CCR7_expression[location_prev == 2] ** n_hill_CCR7_blood_to_ln
        / (
            K_CCR7_blood_to_ln**n_hill_CCR7_blood_to_ln
            + CCR7_expression[location_prev == 2] ** n_hill_CCR7_blood_to_ln
        )
        * CD62L_expression[location_prev == 2] ** n_hill_CD62L_blood_to_ln
        / (
            K_CD62L_blood_to_ln**n_hill_CD62L_blood_to_ln
            + CD62L_expression[location_prev == 2] ** n_hill_CD62L_blood_to_ln
        )
        * K_S1PR1_blood_to_ln**n_hill_S1PR1_blood_to_ln
        / (
            K_S1PR1_blood_to_ln**n_hill_S1PR1_blood_to_ln
            + S1PR1_expression[location_prev == 2] ** n_hill_S1PR1_blood_to_ln
        )
        * 1
        / (
            1
            + np.exp(
                steepness
                * (Lambda_expression[location_prev == 2] - K_Lambda_blood_to_ln)
            )
        )
        * dt
    )

    return transition_rate


def ln_to_blood_transition(
    location_prev,
    S1PR1_expression,
    CCR7_expression,
    ln_to_blood_rate,
    K_S1PR1_ln_to_blood,
    K_CCR7_ln_to_blood,
    n_hill_S1PR1_ln_to_blood,
    n_hill_CCR7_ln_to_blood,
    dt,
):
    """
    Calculate the transition of cells from lymph nodes to blood based on S1PR1 expression.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - S1PR1_expression: Expression values for S1PR1, should be shape (n_cells,).
    - ln_to_blood_rate: Rate of transition from lymph nodes to blood.
    - K_S1PR1_ln_to_blood: Half-saturation constant for S1PR1 in the transition.
    - n_hill_S1PR1_ln_to_blood: Hill coefficient for S1PR1 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    CCR7_expression = CCR7_expression.astype(np.int64)

    transition_rate = (
        ln_to_blood_rate
        * S1PR1_expression[location_prev == 1] ** n_hill_S1PR1_ln_to_blood
        / (
            K_S1PR1_ln_to_blood**n_hill_S1PR1_ln_to_blood
            + S1PR1_expression[location_prev == 1] ** n_hill_S1PR1_ln_to_blood
        )
        * 5
        * K_CCR7_ln_to_blood**n_hill_CCR7_ln_to_blood
        / (
            K_CCR7_ln_to_blood**n_hill_CCR7_ln_to_blood
            + CCR7_expression[location_prev == 1] ** n_hill_CCR7_ln_to_blood
        )
        * dt
    )
    return transition_rate


def blood_to_spleen_transition(
    location_prev,
    CCR7_expression,
    S1PR1_expression,
    Lambda_expression,
    blood_to_spleen_rate,
    K_CCR7_blood_to_spleen,
    K_S1PR1_blood_to_spleen,
    K_Lambda_blood_to_spleen,
    n_hill_CCR7_blood_to_spleen,
    n_hill_S1PR1_blood_to_spleen,
    steepness,
    dt,
):
    """
    Calculate the transition of cells from blood to spleen.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - CCR7_expression: Expression values for CCR7, should be shape (n_cells,).
    - blood_to_spleen_rate: Rate of transition from blood to spleen.
    - K_CCR7_blood_to_spleen: Half-saturation constant for CCR7 in the transition.
    - n_hill_CCR7_blood_to_spleen: Hill coefficient for CCR7 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    CCR7_expression = CCR7_expression.astype(np.int64)
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    Lambda_expression = Lambda_expression.astype(np.int64)

    transition_rate = (
        blood_to_spleen_rate
        * CCR7_expression[location_prev == 2] ** n_hill_CCR7_blood_to_spleen
        / (
            K_CCR7_blood_to_spleen**n_hill_CCR7_blood_to_spleen
            + CCR7_expression[location_prev == 2] ** n_hill_CCR7_blood_to_spleen
        )
        * K_S1PR1_blood_to_spleen**n_hill_S1PR1_blood_to_spleen
        / (
            K_S1PR1_blood_to_spleen**n_hill_S1PR1_blood_to_spleen
            + S1PR1_expression[location_prev == 2] ** n_hill_S1PR1_blood_to_spleen
        )
        * 1
        / (
            1
            + np.exp(
                steepness
                * (Lambda_expression[location_prev == 2] - K_Lambda_blood_to_spleen)
            )
        )
        * dt
    )
    return transition_rate


def spleen_to_blood_transition(
    location_prev,
    S1PR1_expression,
    spleen_to_blood_rate,
    K_S1PR1_spleen_to_blood,
    n_hill_S1PR1_spleen_to_blood,
    dt,
):
    """
    Calculate the transition of cells from spleen to blood.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - S1PR1_expression: Expression values for S1PR1, should be shape (n_cells,).
    - spleen_to_blood_rate: Rate of transition from spleen to blood.
    - K_S1PR1_spleen_to_blood: Half-saturation constant for S1PR1 in the transition.
    - n_hill_S1PR1_spleen_to_blood: Hill coefficient for S1PR1 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    S1PR1_expression = S1PR1_expression.astype(np.int64)

    transition_rate = (
        spleen_to_blood_rate
        * S1PR1_expression[location_prev == 3] ** n_hill_S1PR1_spleen_to_blood
        / (
            K_S1PR1_spleen_to_blood**n_hill_S1PR1_spleen_to_blood
            + S1PR1_expression[location_prev == 3] ** n_hill_S1PR1_spleen_to_blood
        )
        * dt
    )
    return transition_rate


def blood_to_BM_transition(
    location_prev,
    CXCR4_expression,
    CD44_expression,
    S1PR1_expression,
    Lambda_expression,
    K_Lambda_blood_to_BM,
    blood_to_BM_rate,
    K_CXCR4_blood_to_BM,
    K_CD44_blood_to_BM,
    K_S1PR1_blood_to_BM,
    n_hill_CXCR4_blood_to_BM,
    n_hill_CD44_blood_to_BM,
    n_hill_S1PR1_blood_to_BM,
    steepness,
    dt,
):
    """
    Calculate the transition of cells from blood to bone marrow.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - CXCR4_expression: Expression values for CXCR4, should be shape (n_cells,).
    - CD44_expression: Expression values for CD44, should be shape (n_cells,).
    - blood_to_BM_rate: Rate of transition from blood to bone marrow.
    - K_CXCR4_blood_to_BM: Half-saturation constant for CXCR4 in the transition.
    - K_CD44_blood_to_BM: Half-saturation constant for CD44 in the transition.
    - n_hill_CXCR4_blood_to_BM: Hill coefficient for CXCR4 in the transition.
    - n_hill_CD44_blood_to_BM: Hill coefficient for CD44 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    CXCR4_expression = CXCR4_expression.astype(np.int64)
    CD44_expression = CD44_expression.astype(np.int64)
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    Lambda_expression = Lambda_expression.astype(np.int64)

    transition_rate = (
        blood_to_BM_rate
        * CXCR4_expression[location_prev == 2] ** n_hill_CXCR4_blood_to_BM
        / (
            K_CXCR4_blood_to_BM**n_hill_CXCR4_blood_to_BM
            + CXCR4_expression[location_prev == 2] ** n_hill_CXCR4_blood_to_BM
        )
        * CD44_expression[location_prev == 2] ** n_hill_CD44_blood_to_BM
        / (
            K_CD44_blood_to_BM**n_hill_CD44_blood_to_BM
            + CD44_expression[location_prev == 2] ** n_hill_CD44_blood_to_BM
        )
        * K_S1PR1_blood_to_BM**n_hill_S1PR1_blood_to_BM
        / (
            K_S1PR1_blood_to_BM**n_hill_S1PR1_blood_to_BM
            + S1PR1_expression[location_prev == 2] ** n_hill_S1PR1_blood_to_BM
        )
        * 1
        / (
            1
            + np.exp(
                steepness
                * (Lambda_expression[location_prev == 2] - K_Lambda_blood_to_BM)
            )
        )
        * dt
    )

    return transition_rate


def BM_to_blood_transition(
    location_prev,
    S1PR1_expression,
    CD44_expression,
    CXCR4_expression,
    BM_to_blood_rate,
    K_S1PR1_BM_to_blood,
    K_CD44_BM_to_blood,
    K_CXCR4_BM_to_blood,
    n_hill_S1PR1_BM_to_blood,
    n_hill_CD44_BM_to_blood,
    n_hill_CXCR4_BM_to_blood,
    dt,
):
    """
    Calculate the transition of cells from bone marrow to blood.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - S1PR1_expression: Expression values for S1PR1, should be shape (n_cells,).
    - BM_to_blood_rate: Rate of transition from bone marrow to blood.
    - K_S1PR1_BM_to_blood: Half-saturation constant for S1PR1 in the transition.
    - K_CD44_BM_to_blood: Half-saturation constant for CD44 in the transition.
    - n_hill_S1PR1_BM_to_blood: Hill coefficient for S1PR1 in the transition.
    - n_hill_CD44_BM_to_blood: Hill coefficient for CD44 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    CD44_expression = CD44_expression.astype(np.int64)
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    CXCR4_expression = CXCR4_expression.astype(np.int64)

    transition_rate = (
        BM_to_blood_rate
        * S1PR1_expression[location_prev == 4] ** n_hill_S1PR1_BM_to_blood
        / (
            K_S1PR1_BM_to_blood**n_hill_S1PR1_BM_to_blood
            + S1PR1_expression[location_prev == 4] ** n_hill_S1PR1_BM_to_blood
        )
        * K_CD44_BM_to_blood**n_hill_CD44_BM_to_blood
        / (
            K_CD44_BM_to_blood**n_hill_CD44_BM_to_blood
            + CD44_expression[location_prev == 4] ** n_hill_CD44_BM_to_blood
        )
        * K_CXCR4_BM_to_blood**n_hill_CXCR4_BM_to_blood
        / (
            K_CXCR4_BM_to_blood**n_hill_CXCR4_BM_to_blood
            + CXCR4_expression[location_prev == 4] ** n_hill_CXCR4_BM_to_blood
        )
        * dt
    )

    return transition_rate


def blood_to_gut_transition(
    location_prev,
    a4b7_expression,
    S1PR1_expression,
    Lambda_expression,
    blood_to_gut_rate,
    K_a4b7_blood_to_gut,
    K_S1PR1_blood_to_gut,
    K_Lambda_blood_to_gut,
    n_hill_a4b7_blood_to_gut,
    n_hill_S1PR1_blood_to_gut,
    steepness,
    dt,
):
    """
    Calculate the transition of cells from blood to gut based on a4b7 expression.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - a4b7_expression: Expression values for a4b7, should be shape (n_cells,).
    - blood_to_gut_rate: Rate of transition from blood to gut.
    - K_a4b7_blood_to_gut: Half-saturation constant for a4b7 in the transition.
    - n_hill_a4b7_blood_to_gut: Hill coefficient for a4b7 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    a4b7_expression = a4b7_expression.astype(np.int64)
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    Lambda_expression = Lambda_expression.astype(np.int64)

    transition_rate = (
        blood_to_gut_rate
        * a4b7_expression[location_prev == 2] ** n_hill_a4b7_blood_to_gut
        / (
            K_a4b7_blood_to_gut**n_hill_a4b7_blood_to_gut
            + a4b7_expression[location_prev == 2] ** n_hill_a4b7_blood_to_gut
        )
        * K_S1PR1_blood_to_gut**n_hill_S1PR1_blood_to_gut
        / (
            K_S1PR1_blood_to_gut**n_hill_S1PR1_blood_to_gut
            + S1PR1_expression[location_prev == 2] ** n_hill_S1PR1_blood_to_gut
        )
        * 1
        / (
            1
            + np.exp(
                steepness
                * (Lambda_expression[location_prev == 2] - K_Lambda_blood_to_gut)
            )
        )
        * dt  # scatterplot x -> expression Lambda y -> n_hill wert hoch
    )

    return transition_rate


def gut_to_blood_transition(
    location_prev,
    S1PR1_expression,
    gut_to_blood_rate,
    K_S1PR1_gut_to_blood,
    n_hill_S1PR1_gut_to_blood,
    dt,
):
    """
    Calculate the transition of cells from gut to blood based on S1PR1 expression.

    Parameters:
    - location_prev: Previous location of cells, should be shape (n_cells,).
    - S1PR1_expression: Expression values for S1PR1, should be shape (n_cells,).
    - gut_to_blood_rate: Rate of transition from gut to blood.
    - K_S1PR1_gut_to_blood: Half-saturation constant for S1PR1 in the transition.
    - n_hill_S1PR1_gut_to_blood: Hill coefficient for S1PR1 in the transition.
    - dt: Time step for the reaction.

    Returns:
    - Updated location array after migration.
    """
    S1PR1_expression = S1PR1_expression.astype(np.int64)
    transition_rate = (
        gut_to_blood_rate
        * S1PR1_expression[location_prev == 5] ** n_hill_S1PR1_gut_to_blood
        / (
            K_S1PR1_gut_to_blood**n_hill_S1PR1_gut_to_blood
            + S1PR1_expression[location_prev == 5] ** n_hill_S1PR1_gut_to_blood
        )
        * dt
    )

    return transition_rate
