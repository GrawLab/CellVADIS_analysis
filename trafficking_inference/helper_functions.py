import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches

marker_dict = {  # Needs to be adjusted if model changed!
    "CD44": 0,
    "CCR7": 1,
    "S1PR1": 2,
    "CXCR4": 3,
    "CD69": 4,
    "CD62L": 5,
    "a4b7": 6,
}


def thresholded_accuracy(
    fate_matrix, true_next_locations, threshold=0.3, fate_groups_ids=(1, 3, 4, 5)
):
    correct = 0
    total = fate_matrix.shape[0]
    for i in range(fate_matrix.shape[0]):
        true_loc = true_next_locations[i]
        if true_loc == 2: #This if statement is never really used, 
            # as cells with true_loc == 2 are discarded directly after simulation
            continue
        fate_index = fate_groups_ids.index(true_loc)
        if fate_matrix[i, fate_index] >= threshold:
            correct += 1
    return correct / total


def avg_thresholded_accuracy(
    fate_matrix, true_next_locations, fate_groups_ids=(1, 3, 4, 5)
):
    thresholds = np.arange(0.0, 1.0, 0.001)
    thres_accs = []
    for threshold in thresholds:
        thres_acc = thresholded_accuracy(
            fate_matrix,
            true_next_locations,
            threshold=threshold,
            fate_groups_ids=fate_groups_ids,
        )
        thres_accs.append(thres_acc)
    return np.mean(thres_accs)


def get_predicted_location_fractions_rawOT(
    OT_plan, target_locations, fate_groups_ids=(1, 3, 4, 5)
):
    predicted_fractions = np.zeros(len(fate_groups_ids))
    for i, fate_id in enumerate(fate_groups_ids):
        predicted_fractions[i] = np.sum(
            OT_plan[:, target_locations == fate_id]
        ) / np.sum(OT_plan)
    return predicted_fractions


def get_predicted_location_fractions(fate_matrix, fate_groups_ids=(1, 3, 4, 5)):
    predicted_fractions = np.zeros(len(fate_groups_ids))
    for i, fate_id in enumerate(fate_groups_ids):
        predicted_fractions[i] = np.sum(fate_matrix[:, i]) / np.sum(fate_matrix)
    return predicted_fractions


def get_predicted_location_fractions_thresholded(
    fate_matrix, fate_groups_ids=(1, 3, 4, 5), threshold=1 / 3
):
    thresholded_fate_matrix = (fate_matrix > threshold).astype(int)
    predicted_fractions = np.zeros(len(fate_groups_ids))
    for i, fate_id in enumerate(fate_groups_ids):
        predicted_fractions[i] = np.sum(thresholded_fate_matrix[:, i]) / np.sum(
            thresholded_fate_matrix
        )
    return predicted_fractions


def get_predicted_location_fractions_argmax(fate_matrix, fate_groups_ids=(1, 3, 4, 5)):
    argmax_fate_matrix = np.zeros_like(fate_matrix)
    argmax_indices = np.argmax(fate_matrix, axis=1)
    argmax_fate_matrix[np.arange(fate_matrix.shape[0]), argmax_indices] = 1
    predicted_fractions = np.zeros(len(fate_groups_ids))
    for i, fate_id in enumerate(fate_groups_ids):
        predicted_fractions[i] = np.sum(argmax_fate_matrix[:, i]) / np.sum(
            argmax_fate_matrix
        )
    return predicted_fractions


def plot_accuracy_vs_threshold(
    fate_matrix_pred,
    fate_matrix_true,
    true_next_locations,
    fate_groups_ids=(1, 3, 4, 5),
    save_path="accuracy_vs_threshold.png",
):
    thresholds = np.arange(0.0, 1.0, 0.001)
    pred_accs = []
    true_accs = []
    for threshold in thresholds:
        pred_acc = thresholded_accuracy(
            fate_matrix_pred,
            true_next_locations,
            threshold=threshold,
            fate_groups_ids=fate_groups_ids,
        )
        true_acc = thresholded_accuracy(
            fate_matrix_true,
            true_next_locations,
            threshold=threshold,
            fate_groups_ids=fate_groups_ids,
        )
        pred_accs.append(pred_acc)
        true_accs.append(true_acc)
    plt.plot(thresholds, pred_accs, label="Predicted")
    plt.plot(thresholds, true_accs, label="True")
    plt.xlabel("Threshold")
    # compute AUC_pred/AUC_true
    auc_pred = np.trapezoid(pred_accs, thresholds)
    auc_true = np.trapezoid(true_accs, thresholds)
    plt.title(f"Accuracy vs Threshold (AUC_pred/AUC_true = {auc_pred / auc_true:.3f})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_integrated_absolute_error(
    fate_matrix_pred,
    fate_matrix_true,
    true_next_locations,
    fate_groups_ids=(1, 3, 4, 5),
):
    thresholds = np.arange(0.0, 1.0, 0.001)
    pred_accs = []
    true_accs = []
    for threshold in thresholds:
        pred_acc = thresholded_accuracy(
            fate_matrix_pred,
            true_next_locations,
            threshold=threshold,
            fate_groups_ids=fate_groups_ids,
        )
        true_acc = thresholded_accuracy(
            fate_matrix_true,
            true_next_locations,
            threshold=threshold,
            fate_groups_ids=fate_groups_ids,
        )
        pred_accs.append(pred_acc)
        true_accs.append(true_acc)
    return np.mean(np.abs(np.array(pred_accs) - np.array(true_accs)))


def get_avg_relative_thresholded_accuracy(
    fate_matrix,
    true_fate_probabilities,
    true_next_locations,
    thresholds=np.arange(0.2, 0.5, 0.05),
):
    rel_thres_acc_list = []
    for thres in thresholds:
        pred_acc = thresholded_accuracy(
            fate_matrix,
            true_next_locations,
            threshold=thres,
        )
        true_acc = thresholded_accuracy(
            true_fate_probabilities,
            true_next_locations,
            threshold=thres,
        )
        rel_thres_acc_list.append(pred_acc / true_acc)
    return np.mean(rel_thres_acc_list)


def get_true_fate_probabilities(data, params, with_unobserved=False):
    # use functions from migration_functions.py to compute transition probabilities
    # This function depends on the parameters need to compute the transition rates,
    # so if the transition rate in .migration_functions was modified such that different parameters are introduced,
    # then this needs to be accounted for here when calling the respective transition function!
    # same goes for the introduction of other compartments!
    location_prev = np.full(
        data.shape[0], 2
    )  # assume all cells are in blood (location 2)
    if with_unobserved:
        from simulator_unobserved.migration_functions import (
            blood_to_ln_transition,
            blood_to_spleen_transition,
            blood_to_BM_transition,
            blood_to_gut_transition,
            blood_to_U_transition,
        )
        from simulator_unobserved import marker_dict

        transition_probs = np.zeros((data.shape[0], 5))
        transition_probs[:, 0] = blood_to_ln_transition(
            location_prev=location_prev,
            CCR7_expression=data[:, marker_dict["CCR7"]],
            CD62L_expression=data[:, marker_dict["CD62L"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            Lambda_expression=data[:, marker_dict["Lambda"]],
            blood_to_ln_rate=params["blood_to_ln_rate"],
            n_hill_CCR7_blood_to_ln=params["n_hill_CCR7_blood_to_ln"],
            K_CCR7_blood_to_ln=params["K_CCR7_blood_to_ln"],
            n_hill_CD62L_blood_to_ln=params["n_hill_CD62L_blood_to_ln"],
            K_CD62L_blood_to_ln=params["K_CD62L_blood_to_ln"],
            n_hill_S1PR1_blood_to_ln=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_ln=params["K_S1PR1_blood_to_xx"],
            K_Lambda_blood_to_ln=params["K_Lambda_blood_to_xx"],
            steepness=params["Lambda_steepness"],
            dt=1,
        )

        transition_probs[:, 1] = blood_to_spleen_transition(
            location_prev=location_prev,
            CCR7_expression=data[:, marker_dict["CCR7"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            Lambda_expression=data[:, marker_dict["Lambda"]],
            blood_to_spleen_rate=params["blood_to_spleen_rate"],
            n_hill_CCR7_blood_to_spleen=params["n_hill_CCR7_blood_to_spleen"],
            K_CCR7_blood_to_spleen=params["K_CCR7_blood_to_spleen"],
            n_hill_S1PR1_blood_to_spleen=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_spleen=params["K_S1PR1_blood_to_xx"],
            K_Lambda_blood_to_spleen=params["K_Lambda_blood_to_xx"],
            steepness=params["Lambda_steepness"],
            dt=1,
        )

        transition_probs[:, 2] = blood_to_BM_transition(
            location_prev=location_prev,
            CXCR4_expression=data[:, marker_dict["CXCR4"]],
            CD44_expression=data[:, marker_dict["CD44"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            Lambda_expression=data[:, marker_dict["Lambda"]],
            blood_to_BM_rate=params["blood_to_BM_rate"],
            n_hill_CXCR4_blood_to_BM=params["n_hill_CXCR4_blood_to_BM"],
            K_CXCR4_blood_to_BM=params["K_CXCR4_blood_to_BM"],
            n_hill_CD44_blood_to_BM=params["n_hill_CD44_blood_to_BM"],
            K_CD44_blood_to_BM=params["K_CD44_blood_to_BM"],
            n_hill_S1PR1_blood_to_BM=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_BM=params["K_S1PR1_blood_to_xx"],
            K_Lambda_blood_to_BM=params["K_Lambda_blood_to_xx"],
            steepness=params["Lambda_steepness"],
            dt=1,
        )

        transition_probs[:, 3] = blood_to_gut_transition(
            location_prev=location_prev,
            a4b7_expression=data[:, marker_dict["a4b7"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            Lambda_expression=data[:, marker_dict["Lambda"]],
            blood_to_gut_rate=params["blood_to_gut_rate"],
            n_hill_a4b7_blood_to_gut=params["n_hill_a4b7_blood_to_gut"],
            K_a4b7_blood_to_gut=params["K_a4b7_blood_to_gut"],
            n_hill_S1PR1_blood_to_gut=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_gut=params["K_S1PR1_blood_to_xx"],
            K_Lambda_blood_to_gut=params["K_Lambda_blood_to_xx"],
            steepness=params["Lambda_steepness"],
            dt=1,
        )

        transition_probs[:, 4] = blood_to_U_transition(
            location_prev=location_prev,
            Lambda_expression=data[:, marker_dict["Lambda"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            blood_to_U_rate=params["blood_to_U_rate"],
            K_Lambda_blood_to_U=params["K_Lambda_blood_to_xx"],
            steepness=params["Lambda_steepness"],
            n_hill_S1PR1_blood_to_U=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_U=params["K_S1PR1_blood_to_xx"],
            dt=1,
        )
    else:
        from simulator.migration_functions import (
            blood_to_BM_transition,
            blood_to_gut_transition,
            blood_to_ln_transition,
            blood_to_spleen_transition,
        )
        from simulator import marker_dict

        # compute transition probabilities
        transition_probs = np.zeros((data.shape[0], 4))
        transition_probs[:, 0] = blood_to_ln_transition(
            location_prev=location_prev,
            CCR7_expression=data[:, marker_dict["CCR7"]],
            CD62L_expression=data[:, marker_dict["CD62L"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            blood_to_ln_rate=params["blood_to_ln_rate"],
            n_hill_CCR7_blood_to_ln=params["n_hill_CCR7_blood_to_ln"],
            K_CCR7_blood_to_ln=params["K_CCR7_blood_to_ln"],
            n_hill_CD62L_blood_to_ln=params["n_hill_CD62L_blood_to_ln"],
            K_CD62L_blood_to_ln=params["K_CD62L_blood_to_ln"],
            n_hill_S1PR1_blood_to_ln=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_ln=params["K_S1PR1_blood_to_xx"],
            dt=1,
        )
        transition_probs[:, 1] = blood_to_spleen_transition(
            location_prev=location_prev,
            CCR7_expression=data[:, marker_dict["CCR7"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            blood_to_spleen_rate=params["blood_to_spleen_rate"],
            n_hill_CCR7_blood_to_spleen=params["n_hill_CCR7_blood_to_spleen"],
            K_CCR7_blood_to_spleen=params["K_CCR7_blood_to_spleen"],
            n_hill_S1PR1_blood_to_spleen=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_spleen=params["K_S1PR1_blood_to_xx"],
            dt=1,
        )

        transition_probs[:, 2] = blood_to_BM_transition(
            location_prev=location_prev,
            CXCR4_expression=data[:, marker_dict["CXCR4"]],
            CD44_expression=data[:, marker_dict["CD44"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            blood_to_BM_rate=params["blood_to_BM_rate"],
            n_hill_CXCR4_blood_to_BM=params["n_hill_CXCR4_blood_to_BM"],
            K_CXCR4_blood_to_BM=params["K_CXCR4_blood_to_BM"],
            n_hill_CD44_blood_to_BM=params["n_hill_CD44_blood_to_BM"],
            K_CD44_blood_to_BM=params["K_CD44_blood_to_BM"],
            n_hill_S1PR1_blood_to_BM=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_BM=params["K_S1PR1_blood_to_xx"],
            dt=1,
        )

        transition_probs[:, 3] = blood_to_gut_transition(
            location_prev=location_prev,
            a4b7_expression=data[:, marker_dict["a4b7"]],
            S1PR1_expression=data[:, marker_dict["S1PR1"]],
            blood_to_gut_rate=params["blood_to_gut_rate"],
            n_hill_a4b7_blood_to_gut=params["n_hill_a4b7_blood_to_gut"],
            K_a4b7_blood_to_gut=params["K_a4b7_blood_to_gut"],
            n_hill_S1PR1_blood_to_gut=params["n_hill_S1PR1_blood_to_xx"],
            K_S1PR1_blood_to_gut=params["K_S1PR1_blood_to_xx"],
            dt=1,
        )

    return transition_probs


def get_fates(ot_matrix, fate_groups):
    # returns a fate matrix of shape (ot_matrix.shape[0], len(fate_groups))
    n_source = ot_matrix.shape[0]
    n_fates = len(fate_groups)
    fate_matrix = np.zeros((n_source, n_fates))
    for i, group in enumerate(fate_groups):
        fate_matrix[:, i] = np.sum(ot_matrix[:, group], axis=1)
    # normalize fate matrix rows to sum to 1
    # fate_matrix = fate_matrix / fate_matrix.sum(axis=1, keepdims=True)
    return fate_matrix


def get_time_till_next_location(data_raw, timepoint):
    next_locations = get_true_next_locations(data_raw, timepoint)
    current_locations = data_raw[timepoint, :, -1]
    time_till_next_location = np.full(next_locations.shape, np.inf)
    for i in range(data_raw.shape[1]):
        current_loc = current_locations[i]
        next_loc = next_locations[i]
        if current_loc != next_loc:
            # find the time it takes for the cell to transition to the next location
            transition_time = np.where(data_raw[:, i, -1] == next_loc)[0]
            transition_time = transition_time[transition_time > timepoint]
            if len(transition_time) > 0:
                time_till_next_location[i] = transition_time[0] - timepoint
    return time_till_next_location


def get_time_since_last_location(data_raw, timepoint):
    current_locations = data_raw[timepoint, :, -1]
    time_since_last_location = np.full(current_locations.shape, np.inf)
    for i in range(data_raw.shape[1]):
        current_loc = current_locations[i]
        # find the last time it was in a different location
        last_time = np.where(data_raw[:, i, -1] != current_loc)[0]
        last_time = last_time[last_time < timepoint]
        if len(last_time) > 0:
            time_since_last_location[i] = timepoint - last_time[-1]
    return time_since_last_location


def plot_fate_probabilities_with_pie_charts(
    ax,
    data_df,
    probability_df,
    entropy_fade=True,
    plot_order=None,
    n_cells_to_plot=1,
    frac=None,
    indices=None,
    s=1,
    outline=1.5,
    background=None,
    colors=None,
    color_map=None,
    locations=None,
    plot_legend=True,
):
    """Plot fate probabilities as pie charts."""
    # adapted from: https://github.com/dahlinlab/MultistageOT.git

    # Compute entropy for each probability distribution

    data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

    if indices is not None:
        data_df_sampled = data_df.loc[indices]
    else:
        data_df_sampled = data_df
    if frac is not None:
        data_df_sampled = data_df_sampled.sample(frac=frac, random_state=42)
    else:
        """
        #instead of randomly sampling cells make sure to sample cells which umap coordinates are not too closer to each other, to avoid overlapping pie charts
        #use sklearn NearestNeighbors to find cells which are too close to each other and remove them from the sample
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data_df_sampled.values)
        distances, indices = nbrs.kneighbors(data_df_sampled.values)
        # Remove cells that are too close to each other
        to_remove = []
        for i in range(len(indices)):
            if distances[i][1] < 0.01:  # Adjust the threshold as needed
                to_remove.append(i)
        data_df_sampled = data_df_sampled.drop(data_df_sampled.index[to_remove])
        """
        data_df_sampled = data_df_sampled.sample(n=n_cells_to_plot, random_state=42)

    # Define colors for each class
    if plot_order is not None:
        probability_df = probability_df[plot_order]

    if colors is None:
        colors = plt.cm.tab10(range(len(probability_df.columns)))

    if entropy_fade:
        ax.scatter(
            data_df_sampled.iloc[:, 0],
            data_df_sampled.iloc[:, 1],
            color=[0.9, 0.9, 0.9],
            s=s,
        )

    # Iterate over each data point
    it = 0
    for i, row in data_df_sampled.iterrows():
        it += 1

        x, y = row.iloc[0], row.iloc[1]
        probs = probability_df.loc[i]
        probs = probs / np.sum(probs)

        # print("\r", "Plotting cell nr {j}/{all}".format(j=it,all=num_data), end="")

        radius = np.sqrt(s) / (2 * 3 * 72)
        ax.set_aspect("equal")
        wedges = [
            patches.Wedge(
                [x, y],
                radius,
                360 * np.sum(probs.iloc[:k]),
                360 * np.sum(probs.iloc[:k]) + 360 * probs.iloc[k],
                linewidth=0,
                color=colors[k],
            )
            for k in range(len(probs))
        ]

        [ax.add_patch(wedges[k]) for k in range(len(probs))]
        if locations is not None and color_map is not None:
            outlines = [
                patches.Wedge(
                    [x, y],
                    r=radius,
                    theta1=0,
                    theta2=360,
                    fill=False,
                    edgecolor=(
                        color_map[int(locations[i])]
                        if locations is not None and color_map is not None
                        else "k"
                    ),
                    linewidth=1,
                )
            ]
            [ax.add_patch(outlines[k]) for k in range(len(outlines))]

        ax.set_ylim(
            np.min(data_df.iloc[:, 1]) - 0.05, np.max(data_df.iloc[:, 1]) + 0.05
        )
        ax.set_xlim(
            np.min(data_df.iloc[:, 0]) - 0.05, np.max(data_df.iloc[:, 0]) + 0.05
        )
        # plot outline of pie chart
        if locations is not None and color_map is not None:
            """
            ax.scatter(
                x,
                y,
                facecolor="none",
                edgecolor=color_map[int(locations[i])] if locations is not None and color_map is not None else "k",
                linewidth=1,
                #s should be equal to the area of the pie chart, which is pi*r^2, so r = sqrt(s/pi)
                s = s/np.pi,
                alpha=1.0,
            )
            """
            # 2. Outline Wedg

        labels = probability_df.columns
        # Create legend patches
        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(colors, labels)
        ]
        # Add the legend
        if plot_legend:
            ax.legend(handles=legend_patches)


def get_true_next_locations(data_raw, timepoint):
    locations_0 = data_raw[timepoint, :, -1]  # take timepoint 0
    locations_over_time_abtp = data_raw[timepoint:, :, -1]
    differences = locations_over_time_abtp - locations_0[None, :]
    # get first non zero difference for each cell after timepoint
    differences = np.argmax(differences != 0, axis=0)
    next_locations = locations_over_time_abtp[
        differences, np.arange(locations_over_time_abtp.shape[1])
    ]
    return next_locations


def get_tv_distance(fate_matrix, true_fate_probabilities):
    """Compute total variation distance between fate matrix and true fate probabilities"""
    # TV distance is defined as 0.5 * sum(|p - q|)
    if true_fate_probabilities.ndim == 1:
        tv_distances = np.mean(np.abs(true_fate_probabilities - fate_matrix))
        # *0.5 not needed here, since we are dealing with a single
        # value and not a distribution over multiple classes summing to 1
    else:
        tv_distances = np.mean(
            0.5 * np.sum(np.abs(true_fate_probabilities - fate_matrix), axis=1)
        )
    return tv_distances


################################
# OLD FUNCTIONS: NOT USED ANYMORE
################################


def get_KL_divergence(fate_matrix, true_fate_probabilities):
    """Compute KL divergence between fate matrix and true fate probabilities"""
    # KL divergence is defined as sum(p * log(p / q))
    # We add a small epsilon to avoid division by zero
    kl_divergences = np.sum(
        true_fate_probabilities
        * np.log(true_fate_probabilities / (fate_matrix + 1e-32) + 1e-32),
        axis=1,
    )
    return kl_divergences


def compute_shortest_path_distance_matrix(source_data, target_data, n_neighbors=15):
    """
    Compute shortest path distance matrix between source and target data.

    Args:
        source_data: Source data matrix (n_source, n_features)
        target_data: Target data matrix (n_target, n_features)
        n_neighbors: Number of neighbors for kNN graph

    Returns:
        shortest_path_matrix: Distance matrix from source to target
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import shortest_path

    combined_data = np.vstack((source_data, target_data))

    # Build kNN graph on combined data
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=1
    ).fit(combined_data)
    distances, indices = nbrs.kneighbors(combined_data)

    # Build adjacency matrix
    n_total = combined_data.shape[0]
    adjacency_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(1, n_neighbors):  # start from 1 to skip self-loop
            adjacency_matrix[i, indices[i, j]] = distances[i, j]
            adjacency_matrix[indices[i, j], i] = distances[i, j]  # undirected graph

    # Compute shortest paths
    shortest_path_matrix = shortest_path(adjacency_matrix, method="D", directed=False)

    # Extract cost matrix from source to target
    cost_matrix = shortest_path_matrix[0 : source_data.shape[0], source_data.shape[0] :]
    cost_matrix = cost_matrix / np.median(cost_matrix)

    return cost_matrix


def train_normalizing_flow(train_data, eval_data, verbose=False):
    # create model
    import normflows as nf
    import torch

    base = nf.distributions.base.DiagGaussian(shape=train_data.shape[1])

    batch_size = 128
    max_epochs = 200

    # Define list of flows

    flows = []
    num_layers = 4
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([4, 64, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(7, mode="swap"))

    model = nf.NormalizingFlow(base, flows)

    # first split train data into train and val
    train_split = 0.9
    val_split = 1 - train_split
    print("Train data shape:", np.shape(train_data))
    n_train = int(len(train_data) * train_split)
    n_val = int(len(train_data) * val_split)
    train_data_split = train_data[:n_train]
    val_data_split = train_data[n_train : n_train + n_val]

    train_loader = torch.utils.data.DataLoader(
        torch.tensor(train_data_split, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=True,
    )
    val_tensor = torch.tensor(val_data_split, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_loss = np.inf
    patience = 10
    patience_counter = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.forward_kld(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = model.forward_kld(val_tensor)
        if epoch % 20 == 0 and verbose:
            print(
                f"Epoch {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(
                        "Early stopping, final epoch:",
                        epoch,
                        "Train Loss:",
                        f"{train_loss:.3f}",
                        "Last Val Loss:",
                        f"{val_loss:.3f}",
                    )
                break

    # Evaluate on eval_data
    model.eval()
    with torch.no_grad():
        eval_tensor = torch.tensor(eval_data, dtype=torch.float32)
        log_probs = model.log_prob(eval_tensor).numpy()
        print(np.min(log_probs), np.max(log_probs))

    # compute ECDF of inverse transformed log probs on train data
    with torch.no_grad():
        tensor_full = torch.tensor(val_data_split, dtype=torch.float32)
        reversed_samples = model.inverse(tensor_full).numpy()
    # check that reversed_samples follow a standard normal distribution
    from scipy import stats

    log_probs_full = stats.norm.logpdf(reversed_samples).sum(axis=1)
    # plot points colored by log probs
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(
        reversed_samples[:, 0], reversed_samples[:, 1], c=log_probs_full, cmap="viridis"
    )
    plt.colorbar(label="Log Probability Density")
    plt.title("Reversed Samples Colored by Log Probability Density")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("reversed_samples_log_probs.png", dpi=300)

    return log_probs


def get_transition_probabilities(data_blood, data_non_blood, locations_non_blood):
    from sklearn.neighbors import KernelDensity

    densities_kde = np.zeros((data_blood.shape[0], 4))
    for i, loc in enumerate([1, 3, 4, 5]):
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(
            data_non_blood[locations_non_blood == loc]
        )
        densities_kde[:, i] = kde.score_samples(data_blood)

    transition_probs = np.exp(densities_kde) / np.sum(
        np.exp(densities_kde), axis=1, keepdims=True
    )
    return transition_probs
