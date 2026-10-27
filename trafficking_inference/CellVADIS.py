import sys
import os

# Add project root to path (for simulator module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Add trafficking_inference to path (for helper_functions)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import accuracy_score
import trafficking_inference.helper_functions as hf
import ot

np.random.seed(42)


def simulate_data(params, procs=24, store_raw=False, with_unobserved=False):
    """
    Simulate cell-trafficking data and prepare a snapshot for OT analysis.

    The simulator first generates trajectories including the transient period. The
    trajectory is then filtered, sampled at the 24-hour snapshot, transformed to
    approximate flow-cytometry measurements, and standardized marker by marker.

    Parameters
    ----------
    params : dict
        Model parameters. Must include ``n_cells`` and ``dt``; the remaining keys
        depend on the selected simulator.
    procs : int, default=24
        Number of worker processes used by the simulator.
    store_raw : bool, default=False
        If True, save the full time-resolved simulation to
        ``simulated_data_raw_overtime.npy`` before snapshot preprocessing.
    with_unobserved : bool, default=False
        If True, use ``simulator_unobserved`` and retain the unobserved
        compartment in the simulated trajectories.

    Returns
    -------
    data_raw : ndarray
        Raw marker expression at the 24-hour snapshot, after removing the
        location column and cells filtered by the blood dwell-time criterion.
    data_normalized : ndarray
        Arcsinh-transformed and marker-wise z-score-standardized expression.
    locations : ndarray
        Snapshot location codes. In the observed model, codes are 1=LN,
        2=Blood, 3=Spleen, 4=BM, and 5=Gut; the unobserved model adds its
        own extra compartment code.
    true_next_locations : ndarray
        First location reached after the snapshot, used as evaluation ground truth.
    """
    # Simulate the full trajectories used to construct the snapshot and ground truth.
    n_cells = int(params["n_cells"])
    dt = params["dt"]
    if with_unobserved:
        from simulator_unobserved.model import multiprocess_simulate_model
    else:
        from simulator.model import multiprocess_simulate_model
    # The benchmark workflow uses a 30-hour post-transient simulation window.
    data_raw = multiprocess_simulate_model(
        params,
        n_cells=n_cells,
        n_cells_max=n_cells,
        n_workers=procs,
        t_end=30,
        dt=dt,
        simulate_trafficking=True,
        return_samples=None,
        use_gpu=False,
    )
    # Filter out cells that never leave the blood compartment (always in location 2)
    data_raw = data_raw[
        :, np.any(data_raw[:, :, -1] != 2, axis=0), :
    ]  # remove cells that are always in blood

    # Select the 24-hour snapshot used by the benchmark workflow.
    timepoint = int(24 / dt)  # timepoint to use for snapshot data: 24 hours
    true_next_locations = hf.get_true_next_locations(data_raw, timepoint)

    # also get the time it takes to get to the next location
    time_till_next_location = hf.get_time_till_next_location(data_raw, timepoint)

    time_since_last_location = hf.get_time_since_last_location(data_raw, timepoint)
    time_in_loc = time_till_next_location + time_since_last_location

    if store_raw:
        np.save(
            "simulated_data_raw_overtime.npy", data_raw
        )  # store raw data for debugging/analysis
    locations = data_raw[timepoint, :, -1]  # take data at timepoint
    data_raw = data_raw[timepoint, :, :-1]
    # Remove cells in blood that have been in blood for more than 6 hours,
    # since these are unlikely to be actively trafficking cells
    blood_time_filter = ~((locations == 2) & (time_in_loc > 6 // dt))
    data_raw = data_raw[blood_time_filter]
    locations = locations[blood_time_filter]
    true_next_locations = true_next_locations[blood_time_filter]
    time_in_loc = time_in_loc[blood_time_filter]

    # Approximate the positive, skewed intensity values of flow-cytometry markers.
    noise_additive = np.random.normal(loc=0, scale=10, size=data_raw.shape)
    data = data_raw**1.8 + noise_additive

    # Arcsinh transformation reduces the dynamic range while preserving zero values.
    data_normalized = np.arcsinh(data / 150)

    # Standardize each marker across the cells in this snapshot.
    data_normalized = (data_normalized - np.mean(data_normalized, axis=0)) / np.std(
        data_normalized, axis=0
    )

    # Historical CD69 gating is retained as disabled code for provenance; it is
    # not part of the current preprocessing pipeline.
    """
    cd69_gating = data_raw[:, marker_dict["CD69"]] <= 24
    data_raw = data_raw[cd69_gating]
    data_normalized = data_normalized[cd69_gating]
    locations = locations[cd69_gating]
    true_next_locations = true_next_locations[cd69_gating]
    time_in_loc = time_in_loc[cd69_gating]
    """
    return data_raw, data_normalized, locations, true_next_locations


def Stitch_OT(
    data_source,
    data_target,
    target_locations,
    target_names,
    tau_a=0.8,
    subsample_non_blood=True,
    rescale_OT_plans=True,
    random_seed=None,
    random_shuffle=False,
    ignore_warnings=True,
    cost_scaling="median",
    epsilon=0.005,
    cost_metric="sqeuclidean",
    assign_unsampled_probs=False,
):
    """
    Infer location-fate probabilities with one unbalanced OT problem per target.

    Target cells are optionally subsampled to the same count in every target
    compartment. Each compartment-specific coupling is computed from the source
    cells and then combined into one coupling matrix.

    Parameters
    ----------
    data_source : ndarray
        Source-cell marker matrix, normally the cells currently in blood.
    data_target : ndarray
        Target-cell marker matrix, normally cells currently outside blood.
        Its rows must correspond to ``target_locations``.
    target_locations : ndarray
        Location label for each row of ``data_target``. Labels must match the
        strings in ``target_names``.
    target_names : sequence of str
        Target location labels and the order of the fate columns in the result.
    tau_a : float, default=0.8
        Source-side unbalanced OT mass parameter. Must be less than 1.
    subsample_non_blood : bool, default=True
        If True, randomly retain the same number of cells per target location,
        equal to the smallest target-location count.
    rescale_OT_plans : bool, default=True
        If True, scale each compartment-specific coupling by its inferred
        fraction before combining the couplings.
    random_seed : int or None, default=None
        Seed for target subsampling and optional coupling shuffling.
    random_shuffle : bool, default=False
        If True, randomly permute all entries of the combined coupling matrix.
        This is intended as a null-model control.
    ignore_warnings : bool, default=True
        If True, suppress warnings globally while this function runs.
    cost_scaling : {"median", "mean"} or None, default="median"
        How to normalize the OT cost matrix. ``None`` leaves the costs unchanged.
    epsilon : float, default=0.005
        Entropic regularization strength used by the unbalanced OT solver.
    cost_metric : str, default="sqeuclidean"
        Distance metric passed to :func:`ot.dist`.
    assign_unsampled_probs : bool, default=False
        If True, append a column estimating the probability that a source cell
        was not represented by any target compartment.

    Returns
    -------
    fate_matrix : ndarray
        Row-normalized source-cell-by-fate probability matrix.
    norm_OT_plan : ndarray
        Row-normalized combined source-to-target coupling matrix.
    OT_plan : ndarray
        Combined, unnormalized source-to-target coupling matrix.
    mask : ndarray of bool
        Boolean mask identifying the selected rows of the original target data.
    """
    if ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")
    if random_seed is not None:
        np.random.seed(random_seed)
    # The target-side mass parameter is fixed for the StitchOT benchmark.
    tau_b = 0.995
    float_type = data_source.dtype
    # Equalize target-cell counts when requested so each location contributes
    # the same number of target observations to its compartment-specific OT.
    min_cells = min(np.unique(target_locations, return_counts=True)[1])
    mask = np.zeros(len(target_locations), dtype=bool)
    for loc in np.unique(target_locations):
        loc_mask = target_locations == loc
        loc_mask = np.asarray(loc_mask).nonzero()[0]
        if subsample_non_blood:
            new_indices = np.random.choice(
                loc_mask, size=min_cells, replace=False
            )
        else:
            new_indices = np.random.choice(
                loc_mask,
                size=int(np.sum(target_locations == loc)),
                replace=False,
            )

        mask[new_indices] = True

    data_target = data_target[mask]
    target_locations = target_locations[mask]

    fate_groups = [np.where(target_locations == loc)[0] for loc in target_names]

    # Compute one source-to-location coupling for each target compartment.
    fate_groups_fractions = np.zeros(len(target_names))
    OT_plan = np.zeros((data_source.shape[0], data_target.shape[0]))
    blood_marginals_threshold = np.zeros((data_source.shape[0], len(fate_groups)))
    for i, loc in enumerate(target_names):
        C = ot.dist(
            data_source, data_target[target_locations == loc], metric=cost_metric
        )
        if cost_scaling == "median":
            C /= np.median(C)  # Normalize to improve numerical stability
        elif cost_scaling == "mean":
            C /= np.mean(C)
        a = np.ones((C.shape[0])) / C.shape[0]
        b = np.ones((C.shape[1])) / C.shape[1]

        OT_plan_loc = ot.sinkhorn_unbalanced(
            a.astype(float_type),
            b.astype(float_type),
            C.astype(float_type),
            reg=epsilon,
            verbose=False,
            log=False,
            reg_m=(
                (epsilon * tau_a) / (1 - tau_a),
                (epsilon * tau_b) / (1 - tau_b),
            ),
            reg_type="entropy",
        )
        a = OT_plan_loc.sum(axis=1)
        blood_marginals_threshold[:, i] = 2 * a < 1 / data_source.shape[0]
        fate_groups_fractions[i] = np.sum(a > 1 / C.shape[0]) / C.shape[0]
        OT_plan[:, target_locations == loc] = OT_plan_loc

    fate_groups_fractions = fate_groups_fractions / np.sum(fate_groups_fractions)
    # Optionally weight each compartment coupling by its inferred mass fraction.
    if rescale_OT_plans:
        for i, loc in enumerate(target_names):
            OT_plan[:, target_locations == loc] *= fate_groups_fractions[i]

    if random_shuffle:
        # Preserve the matrix shape while destroying its biological structure.
        OT_plan = np.random.permutation(OT_plan.flatten()).reshape(OT_plan.shape)

    # Clean numerical failures before normalizing the coupling.
    if np.isnan(OT_plan).any():
        print("OT_plan contains NaN values, replacing with 0")
    OT_plan = np.nan_to_num(OT_plan, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = OT_plan.sum(axis=1, keepdims=True)
    # Zero rows cannot be normalized and are reported for diagnostic purposes.
    if np.any(row_sums == 0):
        zero_row_indices = np.where(row_sums == 0)[0]
        print(
            f"Warning: The following rows in OT_plan have a sum of zero: {zero_row_indices}"
        )
    norm_OT_plan = OT_plan / row_sums
    norm_OT_plan = np.nan_to_num(norm_OT_plan, nan=0.0, posinf=0.0, neginf=0.0)

    fate_matrix = hf.get_fates(OT_plan, fate_groups)
    if assign_unsampled_probs:
        unsampled_prob = (
            np.all(blood_marginals_threshold, axis=1, keepdims=True).astype(float)
            * 1
            / (C.shape[0])
        )
        fate_matrix = np.hstack((fate_matrix, unsampled_prob))

    fate_matrix = np.nan_to_num(fate_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    fate_row_sums = fate_matrix.sum(axis=1, keepdims=True)
    fate_matrix = fate_matrix / fate_row_sums

    return fate_matrix, norm_OT_plan, OT_plan, mask


def Joint_OT(
    data_source,
    data_target,
    target_locations,
    target_names,
    tau_b=0.5,
    tau_a=0.995,
    use_true_fractions=None,
    subsample_non_blood=True,
    ignore_warnings=True,
    random_seed=None,
    cost_scaling="median",
    epsilon=0.005,
    cost_metric="sqeuclidean",
    assign_unsampled_probs=False,
    lambda1=100,
):
    """
    Infer location-fate probabilities with one joint unbalanced OT problem.

    All target compartments are included in one coupling. The target marginal
    can either be uniform across compartments or supplied as known location
    fractions.

    Parameters
    ----------
    data_source : ndarray
        Source-cell marker matrix, normally the cells currently in blood.
    data_target : ndarray
        Target-cell marker matrix. Its rows must correspond to
        ``target_locations``.
    target_locations : ndarray
        Location label for each target row. Labels must match ``target_names``.
    target_names : sequence of str
        Target location labels and the order of the fate columns in the result.
    tau_b : float, default=0.5
        Target-side unbalanced OT mass parameter. Must be less than 1.
    tau_a : float, default=0.995
        Source-side unbalanced OT mass parameter. Must be less than 1.
    use_true_fractions : array-like or None, default=None
        Optional target-location fractions. If provided, one fraction is
        required for each entry in ``target_names``.
    subsample_non_blood : bool, default=True
        If True, randomly retain the same number of target cells per location.
    ignore_warnings : bool, default=True
        If True, suppress warnings globally while this function runs.
    random_seed : int or None, default=None
        Seed used for target subsampling.
    cost_scaling : {"median", "mean"} or None, default="median"
        How to normalize the OT cost matrix. ``None`` leaves the costs unchanged.
    epsilon : float, default=0.005
        Entropic regularization strength used by the unbalanced OT solver.
    cost_metric : str, default="sqeuclidean"
        Distance metric passed to :func:`ot.dist`.
    assign_unsampled_probs : bool, default=False
        If True, append a column estimating unassigned source mass.
    lambda1 : float, default=100
        Scale factor used when converting unassigned OT mass to the optional
        unsampled probability column.

    Returns
    -------
    fate_matrix : ndarray
        Row-normalized source-cell-by-fate probability matrix.
    norm_OT_plan : ndarray
        Row-normalized source-to-target coupling matrix.
    OT_plan : ndarray
        Unnormalized source-to-target coupling matrix.
    mask : ndarray of bool
        Boolean mask identifying the selected rows of the original target data.
    """
    if ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")
    if random_seed is not None:
        np.random.seed(random_seed)
    float_type = data_source.dtype
    # Equalize target-cell counts when requested.
    min_cells = min(np.unique(target_locations, return_counts=True)[1])
    mask = np.zeros(len(target_locations), dtype=bool)
    for loc in np.unique(target_locations):
        loc_mask = target_locations == loc
        loc_mask = np.asarray(loc_mask).nonzero()[0]
        if subsample_non_blood:
            new_indices = np.random.choice(
                loc_mask, size=min_cells, replace=False
            )
        else:
            new_indices = np.random.choice(
                loc_mask,
                size=int(np.sum(target_locations == loc)),
                replace=False,
            )

        mask[new_indices] = True

    data_target = data_target[mask]
    target_locations = target_locations[mask]

    fate_groups = [np.where(target_locations == loc)[0] for loc in target_names]

    # Initialize uniform source and target marginals.
    a = np.ones(data_source.shape[0]) / data_source.shape[0]
    b = np.ones(data_target.shape[0]) / data_target.shape[0]

    # Replace the target marginal with equal or user-supplied location fractions.
    for j, loc in enumerate(target_names):
        loc_mask = target_locations == loc
        marginal = np.sum(loc_mask)
        if use_true_fractions is not None:
            b[loc_mask] = use_true_fractions[j] / marginal
        else:
            b[loc_mask] = (1 / len(target_names)) / marginal

    # Compute marker-space costs between every source and selected target cell.
    C = ot.dist(data_source, data_target, metric=cost_metric)

    if cost_scaling == "median":
        C /= np.median(C)  # Normalize to improve numerical stability
    elif cost_scaling == "mean":
        C /= np.mean(C)
    # Solve the joint unbalanced OT problem.
    OT_plan = ot.sinkhorn_unbalanced(
        a.astype(float_type),
        b.astype(float_type),
        C.astype(float_type),
        reg=epsilon,
        numItermax=100000,
        verbose=False,
        log=False,
        stopThr=1e-10,
        reg_m=(
            (epsilon * tau_a) / (1 - tau_a),
            (epsilon * tau_b) / (1 - tau_b),
        ),
        reg_type="entropy",
    )

    # Normalize each source row to obtain a conditional target distribution.
    norm_OT_plan = OT_plan / OT_plan.sum(axis=1, keepdims=True)

    # Aggregate target-cell mass into one fate column per location.
    fate_matrix = hf.get_fates(OT_plan, fate_groups)
    if assign_unsampled_probs:
        diff_marginals = np.clip(
            (1 / C.shape[0] - np.sum(OT_plan, axis=1)) * lambda1, 0, 1
        )
        unsampled_prob = diff_marginals.reshape(-1, 1)
        fate_matrix = np.hstack((fate_matrix, unsampled_prob))

    fate_matrix = fate_matrix / np.sum(fate_matrix, axis=1, keepdims=True)

    return fate_matrix, norm_OT_plan, OT_plan, mask


def groundtruth_check(
    fate_matrix,
    OT_plan,
    data_source_raw,
    target_cells_location_ids,
    target_names,
    params,
    true_next_locations,
    fate_groups_ids=(1, 3, 4, 5),  # LN, Spleen, BM, Gut
    with_unobserved=False,
):
    """
    Compare inferred fate probabilities with simulator-derived ground truth.

    Parameters
    ----------
    fate_matrix : ndarray
        Inferred source-cell-by-fate probability matrix.
    OT_plan : ndarray
        Unnormalized source-to-target coupling used to obtain ``fate_matrix``.
    data_source_raw : ndarray
        Raw source-cell marker matrix used to compute model transition
        probabilities.
    target_cells_location_ids : ndarray
        Location label for every target column represented by ``OT_plan``.
    target_names : sequence
        Target labels in the order used by ``OT_plan`` and its fate groups.
    params : dict
        Simulator parameters required by
        :func:`trafficking_inference.helper_functions.get_true_fate_probabilities`.
    true_next_locations : ndarray
        Ground-truth next location for each source cell.
    fate_groups_ids : sequence of int, default=(1, 3, 4, 5)
        Location codes corresponding to the fate columns.
    with_unobserved : bool, default=False
        If True, compute ground truth with the unobserved simulator variant.

    Returns
    -------
    ndarray
        A 16-element vector containing accuracy, thresholded accuracy, total
        variation distances, integrated absolute error, and the random baseline
        described by the output ordering below.
    """

    max_fates = np.argmax(fate_matrix, axis=1)
    predicted_locations = np.array([fate_groups_ids[i] for i in max_fates])
    # Compare argmax predictions with the observed next-location labels.

    # Compute model-based transition probabilities for the same source cells.
    true_fate_probabilities = hf.get_true_fate_probabilities(
        data_source_raw, params, with_unobserved=with_unobserved
    )
    true_fate_probabilities = true_fate_probabilities / true_fate_probabilities.sum(
        axis=1, keepdims=True
    )

    max_fates_true = np.argmax(true_fate_probabilities, axis=1)
    predicted_locations_true = np.array([fate_groups_ids[i] for i in max_fates_true])

    avg_thres_acc_true = hf.avg_thresholded_accuracy(
        true_fate_probabilities,  # [true_next_locations[locations==2]!=2]
        true_next_locations,
        fate_groups_ids=fate_groups_ids,
    )

    avg_thres_acc_pred = hf.avg_thresholded_accuracy(
        fate_matrix,  # [true_next_locations[locations==2]!=2]
        true_next_locations,
        fate_groups_ids=fate_groups_ids,
    )
    rel_avg_thres_acc = avg_thres_acc_pred / avg_thres_acc_true

    accuracy_true = accuracy_score(true_next_locations, predicted_locations_true)
    accuracy_pred = accuracy_score(true_next_locations, predicted_locations)
    rel_acc = accuracy_pred / accuracy_true

    # Compare predicted and observed location fractions using several summaries.
    tv_distances = []

    # The fraction vector follows the order in ``fate_groups_ids``.
    real_next_location_fractions = np.zeros(np.shape(fate_matrix)[1])
    for i, loc in enumerate(fate_groups_ids):
        real_next_location_fractions[i] = np.sum(true_next_locations == loc) / len(
            true_next_locations
        )
    predicted_location_fractions_rawOT = hf.get_predicted_location_fractions_rawOT(
        OT_plan,
        target_locations=target_cells_location_ids,
        fate_groups_ids=target_names,
    )
    predicted_location_fractions = hf.get_predicted_location_fractions(
        fate_matrix, fate_groups_ids
    )
    predicted_location_fractions_thresholded = (
        hf.get_predicted_location_fractions_thresholded(
            fate_matrix, fate_groups_ids, threshold=1 / 3
        )
    )
    predicted_location_fractions_thresholded_50 = (
        hf.get_predicted_location_fractions_thresholded(
            fate_matrix, fate_groups_ids, threshold=1 / 2
        )
    )
    predicted_location_fractions_argmax = hf.get_predicted_location_fractions_argmax(
        fate_matrix, fate_groups_ids
    )

    # Compare raw, continuous, thresholded, argmax, and random fraction estimates.
    tv_distance_fractions_rawOT = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions_rawOT)
    )
    tv_distances = np.append(tv_distances, tv_distance_fractions_rawOT)

    tv_distance_fractions = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions)
    )
    tv_distances = np.append(tv_distances, tv_distance_fractions)

    tv_distance_fractions_thresholded = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions_thresholded)
    )
    tv_distances = np.append(tv_distances, tv_distance_fractions_thresholded)

    tv_distance_fractions_thresholded_50 = 0.5 * np.sum(
        np.abs(
            real_next_location_fractions - predicted_location_fractions_thresholded_50
        )
    )
    tv_distances = np.append(tv_distances, tv_distance_fractions_thresholded_50)

    tv_distance_argmax = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions_argmax)
    )
    tv_distances = np.append(tv_distances, tv_distance_argmax)

    random_predicted_next_location_fractions = np.random.dirichlet(
        np.ones(len(fate_groups_ids)), size=1
    )[0]
    tv_distance_random = 0.5 * np.sum(
        np.abs(real_next_location_fractions - random_predicted_next_location_fractions)
    )
    tv_distances = np.append(tv_distances, tv_distance_random)

    tv_distances_true_probs = []

    predicted_location_fractions = hf.get_predicted_location_fractions(
        true_fate_probabilities, fate_groups_ids
    )
    predicted_location_fractions_thresholded = (
        hf.get_predicted_location_fractions_thresholded(
            true_fate_probabilities, fate_groups_ids, threshold=1 / 3
        )
    )
    predicted_location_fractions_thresholded_50 = (
        hf.get_predicted_location_fractions_thresholded(
            true_fate_probabilities, fate_groups_ids, threshold=1 / 2
        )
    )
    predicted_location_fractions_argmax = hf.get_predicted_location_fractions_argmax(
        true_fate_probabilities, fate_groups_ids
    )
    tv_distance_fractions = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions)
    )
    tv_distances_true_probs = np.append(tv_distances_true_probs, tv_distance_fractions)

    tv_distance_fractions_thresholded = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions_thresholded)
    )
    tv_distances_true_probs = np.append(
        tv_distances_true_probs, tv_distance_fractions_thresholded
    )

    tv_distance_fractions_thresholded_50 = 0.5 * np.sum(
        np.abs(
            real_next_location_fractions - predicted_location_fractions_thresholded_50
        )
    )
    tv_distances_true_probs = np.append(
        tv_distances_true_probs, tv_distance_fractions_thresholded_50
    )

    tv_distance_argmax = 0.5 * np.sum(
        np.abs(real_next_location_fractions - predicted_location_fractions_argmax)
    )
    tv_distances_true_probs = np.append(tv_distances_true_probs, tv_distance_argmax)

    IAE = [
        hf.get_integrated_absolute_error(
            fate_matrix, true_fate_probabilities, true_next_locations, fate_groups_ids
        )
    ]

    tv_distance_probs = [hf.get_tv_distance(fate_matrix, true_fate_probabilities)]
    accuracies = np.array(
        [accuracy_true, avg_thres_acc_true, rel_acc, rel_avg_thres_acc]
    )
    # Concatenate the metric groups into the documented 16-element result.
    """
        The final array entries have the following order:
        0: accuracy_true;
        Absolute accuracy of predicted next locations using true fate probabilities

        1: avg_thres_acc_true;
        Average thresholded accuracy of predicted next locations using true fate probabilities

        2: rel_acc;
        Relative accuracy of predicted next locations using inferred fate matrix compared to true fate probabilities

        3: rel_avg_thres_acc;
        Relative average thresholded accuracy of predicted next locations using inferred fate matrix compared to true fate probabilities

        4: tv_distance_transition_proportions_true_probs;
        Total variation distance between true next location fractions and
        predicted next location fractions using true fate probabilities

        5: tv_distance_transition_proportions_thres33_true_probs;
        Total variation distance between true next location fractions and
          predicted next location fractions using true fate probabilities with thresholding 33%

        6: tv_distance_transition_proportions_thres50_true_probs;
        Total variation distance between true next location fractions and
        predicted next location fractions using true fate probabilities with thresholding 50%

        7: tv_distance_transition_proportions_argmax_true_probs;
        Total variation distance between true next location fractions and
        predicted next location fractions using argmax of true fate probabilities

        8: tv_distance_fractions_rawOT;
        Total variation distance between true next location fractions and
        predicted next location fractions using raw OT plan

        9: tv_distance_fractions;
        Total variation distance between true next location fractions and
        predicted next location fractions using inferred fate matrix

        10: tv_distance_fractions_thresholded_33;
        Total variation distance between true next location fractions and
        predicted next location fractions using the inferred fate matrix with
        thresholding 33%

        11: tv_distance_fractions_thresholded_50;
        Total variation distance between true next location fractions and
        predicted next location fractions using the inferred fate matrix with
        thresholding 50%

        12: tv_distance_argmax;
        Total variation distance between true next location fractions and
        predicted next location fractions using argmax of the inferred fate matrix

        13: tv_distance_random;
        Total variation distance between true next location fractions and
        predicted next location fractions using random predictions

        14: tv_distance_probs;
        mean Total variation distance between inferred fate matrix and true fate probabilities

        15: IAE;
        Integrated absolute error between inferred fate matrix and true fate probabilities
        """
    stats = np.concatenate(
        (accuracies, tv_distances_true_probs, tv_distances, tv_distance_probs, IAE)
    )

    return stats


def groundtruth_check_unobserved(
    fate_matrix,
    data_source_raw,
    params,
    true_next_locations,
    fate_groups_ids=(1, 3, 4, 5, 6),
):
    """
    Evaluate fate probabilities when one destination is unobserved.

    The full metrics compare all four observed compartments plus the unobserved
    compartment. Conditional metrics remove cells whose true next location is
    the unobserved compartment and renormalize the remaining four fates.

    Parameters
    ----------
    fate_matrix : ndarray
        Inferred source-cell-by-fate matrix with four observed columns or five
        columns including the unobserved compartment.
    data_source_raw : ndarray
        Raw source-cell marker matrix used to compute simulator transition
        probabilities.
    params : dict
        Parameters required by the unobserved simulator transition functions.
    true_next_locations : ndarray
        Ground-truth next location for each source cell. Code 6 denotes the
        unobserved compartment.
    fate_groups_ids : sequence of int, default=(1, 3, 4, 5, 6)
        Location codes corresponding to the full fate-column order.

    Returns
    -------
    ndarray
        A 12-element vector containing full and conditional error metrics,
        followed by the inferred fraction for each fate.
    """
    true_fate_probabilities = hf.get_true_fate_probabilities(
        data_source_raw, params, with_unobserved=True
    )
    true_fate_probabilities = true_fate_probabilities / true_fate_probabilities.sum(
        axis=1, keepdims=True
    )

    if fate_matrix.shape[1] == 5:
        full_fate_matrix = fate_matrix
    elif fate_matrix.shape[1] == 4:
        # Methods without an explicit unobserved probability return four columns;
        # append a zero-valued unobserved column so all metrics use five columns.
        full_fate_matrix = np.hstack((fate_matrix, np.zeros((fate_matrix.shape[0], 1))))
    else:
        raise ValueError(
            f"Expected fate_matrix with 4 or 5 columns, got shape {fate_matrix.shape}"
        )

    full_fate_matrix = full_fate_matrix / full_fate_matrix.sum(axis=1, keepdims=True)

    # Conditional metrics exclude the unobserved column and renormalize.
    conditional_true_fate_probabilities = true_fate_probabilities[:, :4]
    conditional_true_fate_probabilities = (
        conditional_true_fate_probabilities
        / conditional_true_fate_probabilities.sum(axis=1, keepdims=True)
    )
    conditional_fate_matrix = full_fate_matrix[:, :4]
    conditional_fate_matrix = conditional_fate_matrix / conditional_fate_matrix.sum(
        axis=1, keepdims=True
    )

    full_avg_thres_acc_true = hf.avg_thresholded_accuracy(
        true_fate_probabilities,
        true_next_locations,
        fate_groups_ids=[1, 3, 4, 5, 6],
    )
    full_avg_thres_acc_pred = hf.avg_thresholded_accuracy(
        full_fate_matrix,
        true_next_locations,
        fate_groups_ids=[1, 3, 4, 5, 6],
    )
    full_rel_avg_thres_acc = full_avg_thres_acc_pred / full_avg_thres_acc_true
    tv_distance_first4 = hf.get_tv_distance(
        full_fate_matrix[:, :4], true_fate_probabilities[:, :4]
    )
    tv_distance_5 = 0.5 * np.mean(
        np.abs(full_fate_matrix[:, 4] - true_fate_probabilities[:, 4])
    )
    full_iae = hf.get_integrated_absolute_error(
        full_fate_matrix,
        true_fate_probabilities,
        true_next_locations,
        [1, 3, 4, 5, 6],
    )

    conditional_mask = true_next_locations != 6
    conditional_true_fate_probabilities = conditional_true_fate_probabilities[
        conditional_mask
    ]
    conditional_fate_matrix = conditional_fate_matrix[conditional_mask]
    conditional_true_next_locations = true_next_locations[conditional_mask]

    conditional_avg_thres_acc_true = hf.avg_thresholded_accuracy(
        conditional_true_fate_probabilities,
        conditional_true_next_locations,
        fate_groups_ids=[1, 3, 4, 5],
    )
    conditional_avg_thres_acc_pred = hf.avg_thresholded_accuracy(
        conditional_fate_matrix,
        conditional_true_next_locations,
        fate_groups_ids=[1, 3, 4, 5],
    )
    conditional_rel_avg_thres_acc = (
        conditional_avg_thres_acc_pred / conditional_avg_thres_acc_true
    )
    conditional_tv_distance = hf.get_tv_distance(
        conditional_fate_matrix, conditional_true_fate_probabilities
    )
    conditional_iae = hf.get_integrated_absolute_error(
        conditional_fate_matrix,
        conditional_true_fate_probabilities,
        conditional_true_next_locations,
        [1, 3, 4, 5],
    )

    predicted_fractions = hf.get_predicted_location_fractions(
        full_fate_matrix, [1, 3, 4, 5, 6]
    )

    """
    The returned entries have the following order:
    0: tv_distance_first4;
        Total variation distance between inferred fate matrix and true fate probabilities for the first 4 compartments (LN, Spleen, BM, Gut)
    1: tv_distance_5;
        Total variation distance between inferred fate matrix and true fate probabilities for the unobserved compartment
    2: full_rel_avg_thres_acc;
        Relative average thresholded accuracy of predicted next locations using inferred fate matrix compared to true fate probabilities
    3: full_iae;
        Integrated absolute error between inferred fate matrix and true fate probabilities for all compartments
    4: conditional_tv_distance;
        Total variation distance between inferred fate matrix and
        true fate probabilities for the first 4 compartments (LN, Spleen, BM, Gut)
        conditioned on cells that do not go to unobserved compartment

    5: conditional_rel_avg_thres_acc;
        Relative average thresholded accuracy of predicted next locations using inferred fate matrix compared to true fate probabilities for the first 4 compartments (LN, Spleen, BM, Gut)
        conditioned on cells that do not go to unobserved compartment
    6: conditional_iae;
        Integrated absolute error between inferred fate matrix and true fate probabilities for the first 4 compartments (LN, Spleen, BM, Gut) conditioned on cells that do not go to unobserved compartment
    7-11: predicted_fractions;
        Predicted fractions of cells going to each compartment (LN, Spleen, BM, Gut, Unobserved) based on the inferred fate matrix
    """
    return np.concatenate(
        (
            np.array(
                [
                    # Global metrics
                    tv_distance_first4,
                    tv_distance_5,
                    full_rel_avg_thres_acc,
                    full_iae,
                    # Conditional metrics
                    conditional_tv_distance,
                    conditional_rel_avg_thres_acc,
                    conditional_iae,
                ]
            ),
            predicted_fractions,
        )
    )
