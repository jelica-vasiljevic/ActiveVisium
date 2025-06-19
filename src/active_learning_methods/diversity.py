from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py


def diversity_sampling_based_on_pretrained_model_representations(predictions_df, hdf5_file_path, total_number_of_samples_for_all_clusters, device='cuda'):
    """
    Selects a set of barcodes based on K-means clustering of representations 
    extracted from a pretrained foundational model (e.g., UNI).

    This method groups the morphological features into clusters and selects the most 
    representative barcode (i.e., the one closest to each cluster centroid).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame indexed by barcodes, containing barcode identifiers and model predictions.
    
    hdf5_file_path : str
        Path to the HDF5 file containing pretrained representations.
    
    total_number_of_samples_for_all_clusters : int
        Total number of samples to select, which is also used as the number of clusters.
    
    device : str
        Device to use for computations (default: 'cuda'). Currently unused but included 
        for compatibility with future extensions.

    Returns
    -------
    barcodes_selected : list of str
        List of selected barcode identifiers, one per cluster.
    
    clusters : pd.DataFrame
        DataFrame indexed by barcodes, containing the assigned cluster ID for each.
    """

    # Load precomputed representations in a single batch
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        hdf5_file_group = hdf5_file['valid']

        # Preallocate a numpy array for storing features
        barcodes = predictions_df.index
        num_barcodes = len(barcodes)
        sample_rep = hdf5_file_group["patch_" + barcodes[0] + ".png"]['rep'][:]
        total_number_of_features = sample_rep.size

        morphological_features = np.zeros((num_barcodes, total_number_of_features))

        for i, barcode in enumerate(tqdm(barcodes)):
            barcode_rep_name = "patch_" + barcode + ".png"
            representation = hdf5_file_group[barcode_rep_name]['rep'][:]
            morphological_features[i] = representation

    # K-means clustering
    num_clusters = total_number_of_samples_for_all_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(morphological_features)
    clusters = kmeans.labels_

    barcodes_selected = []
    for cluster_idx in range(num_clusters):
        # Get points in the current cluster
        cluster_points = morphological_features[clusters == cluster_idx]
        centroid = kmeans.cluster_centers_[cluster_idx]

        # Compute distances and select the closest point
        cluster_distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx = np.argmin(cluster_distances)
        barcodes_selected.append(barcodes[np.where(clusters == cluster_idx)[0][closest_idx]])

    # make dataframe with Barcode and its cluster
    clusters = pd.DataFrame(index=barcodes, columns=['cluster'])
    clusters['cluster'] = kmeans.labels_
    
    return barcodes_selected, clusters

def diversity_sampling_based_computed_features(predictions_df, total_number_of_samples_for_all_clusters, device='cuda'):
    """
    Selects a set of barcodes based on K-means clustering of precomputed feature vectors.

    This method uses the 'Features' column in the predictions DataFrame (assumed to be 
    NumPy arrays) to group data into clusters, then selects the most representative 
    barcode from each cluster — i.e., the one closest to the cluster centroid.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame indexed by barcodes, containing a column 'Features' with 
        precomputed feature vectors.
    
    total_number_of_samples_for_all_clusters : int
        Total number of samples to select, also used as the number of clusters.
    
    device : str
        Device string (default: 'cuda'). Currently unused but kept for compatibility.

    Returns
    -------
    barcodes_selected : list of str
        List of selected barcode identifiers, one per cluster.

    clusters : pd.DataFrame
        DataFrame indexed by barcodes, with a 'cluster' column indicating 
        cluster assignment for each barcode.
    """
    features = np.stack(predictions_df['Features'].values)
    barcodes = predictions_df.index
    num_clusters = total_number_of_samples_for_all_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    clusters = kmeans.labels_

    barcodes_selected = []
    for cluster_idx in range(num_clusters):
        # Get points in the current cluster
        cluster_points = features[clusters == cluster_idx]
        centroid = kmeans.cluster_centers_[cluster_idx]

        # Compute distances and select the closest point
        cluster_distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx = np.argmin(cluster_distances)
        barcodes_selected.append(barcodes[np.where(clusters == cluster_idx)[0][closest_idx]])

    # make dataframe with Barcode and its cluster
    clusters = pd.DataFrame(index=barcodes, columns=['cluster'])
    clusters['cluster'] = kmeans.labels_
    
    return barcodes_selected, clusters

def diversity_sampling(diversity_startegy, predictions_df, representation_path, total_number_of_samples_for_all_clusters, device='cuda', adata_path=None, use_pca=True):
    """
    Selects a diverse set of barcodes (spots) based on the specified diversity sampling strategy.

    Parameters:
    -----------
    diversity_startegy : str
        The strategy to use for sampling. Options:
        - "computed_features"
        - "foundational_model_diversity"
        - "random"
        - "top"
    predictions_df : pd.DataFrame
        DataFrame containing model predictions and metadata (indexed by barcode).
    representation_path : str
        Path to stored representations (used with foundational model strategy).
    total_number_of_samples_for_all_clusters : int
        Number of samples to select in total.
    device : str
        Device used for computing embeddings (default: 'cuda').
    adata_path : str, optional
        Path to AnnData object (currently unused).
    use_pca : bool
        Whether to apply PCA (currently unused).

    Returns:
    --------
    barcodes_selected : np.ndarray
        Array of selected barcode identifiers.
    """
    if diversity_startegy == "computed_features":
        # print in red
        print("\033[91m" + "Diversity sampling based on computed features" + "\033[0m")
        barcodes_selected,_= diversity_sampling_based_computed_features(predictions_df, total_number_of_samples_for_all_clusters, device)
    
    elif diversity_startegy == "foundational_model_diversity":
        # print in red
        print("\033[91m" + "Diversity sampling based on foundational model" + "\033[0m")
        barcodes_selected,_ = diversity_sampling_based_on_pretrained_model_representations(predictions_df, representation_path, total_number_of_samples_for_all_clusters, device)
        
    elif diversity_startegy == "random":
        #print in red
        print("\033[91m" + "Diversity sampling based on random" + "\033[0m")
        # shuffle top_sorted_predictions
        predictions_df = predictions_df.sample(frac=1)
        # select top total_number_of_samples_for_all_clusters from top_sorted_predictions
        barcodes_selected = predictions_df.head(total_number_of_samples_for_all_clusters).index.values

    elif diversity_startegy == "top":
        # print in red
        print("\033[91m" + "Diversity sampling based on top" + "\033[0m")
        barcodes_selected = predictions_df.head(total_number_of_samples_for_all_clusters).index.values
    else:
        raise ValueError('diversity_sampling is not specified - should be one of "computed_features", "foundational_model_diversity", "random", or "top"')
    
    return barcodes_selected
    
