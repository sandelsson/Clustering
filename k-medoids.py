import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def k_medoids_clustering(data, initial_medoids, max_iter=100):
    """
    Custom implementation of K-Medoids clustering using pre-defined initial medoids.

    Parameters:
    - data: Original data matrix in binary format (0s and 1s) or with decimals.
    - initial_medoids: List of indices to use as the initial medoids.
    - max_iter: Maximum number of iterations.

    Returns:
    - data: DataFrame with assigned cluster labels.
    - medoids: Indices of the medoids.
    - labels: Cluster labels for each point.
    """
    # Compute the Dice (Rogerstanimoto) distance matrix as in your original code
    distance_matrix = pairwise_distances(data.values, metric='euclidean')

    # Number of data points
    n_points = distance_matrix.shape[0]
    medoids = initial_medoids[:]
    labels = [-1] * n_points

    # K-Medoids Clustering Loop
    for iteration in range(max_iter):
        # Assign points to nearest medoid
        for i in range(n_points):
            distances_to_medoids = [distance_matrix[i][m] for m in medoids]
            labels[i] = np.argmin(distances_to_medoids)

        # Update medoids
        new_medoids = medoids[:]
        for k in range(len(medoids)):
            # Get all points assigned to this medoid
            cluster_points = [i for i in range(n_points) if labels[i] == k]
            if not cluster_points:
                continue

            # Find the new medoid with minimum total distance to other points in the cluster
            min_distance_sum = float('inf')
            for point in cluster_points:
                distance_sum = sum(distance_matrix[point][other] for other in cluster_points)
                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    new_medoids[k] = point

        # Check for convergence
        if new_medoids == medoids:
            break
        else:
            medoids = new_medoids[:]

    # Assign final cluster labels
    for i in range(n_points):
        distances_to_medoids = [distance_matrix[i][m] for m in medoids]
        labels[i] = np.argmin(distances_to_medoids)

    # Add cluster labels to the original data
    data['Cluster'] = labels

    # Calculate the Silhouette Score
    if len(set(labels)) > 1:  # Ensure more than one cluster for silhouette score calculation
        silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
        print(f"Silhouette Score for K-Medoids clustering: {silhouette_avg:.2f}")
    else:
        print("Silhouette Score is not applicable (only one cluster present).")

    # Calculate the Davies-Bouldin Index
    if len(set(labels)) > 1:
        dbi_score = davies_bouldin_score(data.drop('Cluster', axis=1).values, labels)
        print(f"Davies-Bouldin Index for K-Medoids clustering: {dbi_score:.2f}")
    else:
        print("Davies-Bouldin Index is not applicable (only one cluster present).")

    # NEW: Visualize clusters in 2D using PCA
    visualize_clusters_pca(data, labels, medoids)

    return data, medoids, labels

def visualize_clusters_pca(data, labels, medoids):
    """
    Visualize the clustered data using PCA in 2D, highlighting the clusters and medoids.
    """
    # Extract features (exclude the 'Cluster' column)
    feature_data = data.drop('Cluster', axis=1)

    # Standardize features for PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)

    # Run PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)

    # Create a DataFrame for plotting
    coords_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=data.index)

    # Plot points colored by cluster
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        cluster_points = coords_df[labels == lab]
        plt.scatter(cluster_points['PC1'], cluster_points['PC2'], label=f"Cluster {lab}", alpha=0.7)

    # Highlight medoids
    plt.scatter(coords_df.iloc[medoids]['PC1'], coords_df.iloc[medoids]['PC2'], 
                c='red', marker='X', s=200, label='Medoids')

    plt.title('K-Medoids Clustering (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_technical_cluster_centers(data):
    """
    Compute the technical cluster centers (mode of feature vectors) for each cluster.
    """
    # Exclude the 'Cluster' column to compute modes on feature columns only
    feature_columns = data.columns.drop('Cluster')
    cluster_centers = pd.DataFrame(columns=feature_columns)

    # Compute the mode for each feature in each cluster
    for cluster_label in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_label][feature_columns]
        # Calculate mode along each column
        mode_values = cluster_data.mode().iloc[0]
        cluster_centers = cluster_centers._append(mode_values, ignore_index=True)

    cluster_centers.index = sorted(data['Cluster'].unique())
    cluster_centers.index.name = 'Cluster'

    return cluster_centers.astype(int, errors='ignore')

def generate_cluster_summary(data):
    """
    Generate a summary table for clusters.
    """
    feature_columns = data.columns.drop('Cluster')
    summary_list = []

    for cluster_label in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_label]
        n_companies = cluster_data.shape[0]
        feature_percentages = (cluster_data[feature_columns].mean() * 100).round(2)
        summary = {'Cluster': cluster_label, 'n_companies': n_companies}
        summary.update(feature_percentages.to_dict())
        summary_list.append(summary)

    summary_df = pd.DataFrame(summary_list)
    return summary_df

def list_cluster_members(data):
    """
    List the members (service names) of each cluster.
    """
    cluster_members = {}
    for cluster_label in sorted(data['Cluster'].unique()):
        members = data[data['Cluster'] == cluster_label].index.tolist()
        cluster_members[cluster_label] = members
        print(f"\nCluster {cluster_label} Members:")
        for member in members:
            print(f"- {member}")
    return cluster_members

def visualize_clusters_separately(summary_df):
    """
    Visualize attribute values for each cluster in separate figures.
    The summary_df should include the following columns:
      - 'Cluster': Cluster identifier.
      - 'n_companies': Number of companies in the cluster.
      - Additional columns for each attribute (with value percentages).
    """
    # Determine the columns that represent attribute values (exclude 'Cluster' and 'n_companies')
    attribute_columns = [col for col in summary_df.columns if col not in ['Cluster', 'n_companies']]
    
    # Iterate over each cluster (each row in the summary DataFrame)
    for idx, row in summary_df.iterrows():
        cluster_id = row['Cluster']
        # Create a new figure for the current cluster
        plt.figure(figsize=(6, 4))
        # Extract attribute values for this cluster
        feature_data = row[attribute_columns]
        ax = feature_data.plot(kind='bar', color='skyblue')
        ax.set_ylim(0, 100)
        
        # Format the y-axis tick labels to append a '%' symbol
        ticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(t)}%' for t in ticks])
        
        # Set x-axis tick labels with a slight rotation for clarity
        ax.set_xticklabels(feature_data.index, rotation=45, ha='right')
        
        # Optional: you can add a title if you want, e.g., "Cluster {cluster_id}" or leave it blank
        # plt.title(f"Cluster {cluster_id}")
        
        plt.tight_layout()
        plt.show()

    


# Example Usage
def main():
    # Load the Excel file
    data = pd.read_excel('Freemium list.xlsx', sheet_name='Sheet13')

    # Set 'Service Name' as the index
    data.set_index("Service Name", inplace=True)

    # Convert all feature columns to numeric values, replacing errors with NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill any NaN values with 0 (assuming missing values indicate absence)
    data = data.fillna(0)

    # Convert all non-zero values to 1 (binary conversion)
    #data = data.astype(bool).astype(int)

    # Use the medoids identified from hierarchical clustering

    #initial_medoids = [40, 14, 39, 5, 2, 1] ##alkuperäinen

    #initial_medoids = [14, 21, 27, 3, 4]
    initial_medoids = [33, 27, 21, 4, 3]

    #initial_medoids = [2, 3, 40, 14, 39, 5]
    # Convert medoids to standard Python integers for better readability
    converted_medoids = [int(medoid) for medoid in initial_medoids]
    print("Initial medoids based on hierarchical clustering:", converted_medoids)

    # Perform K-Medoids clustering using initial medoid seeds and calculate validation metrics
    clustered_data, medoids, labels = k_medoids_clustering(data, initial_medoids)

    # Compute technical cluster centers (mode)
    cluster_centers = compute_technical_cluster_centers(clustered_data)
    print("\nTechnical Cluster Centers (Binary Mode):")
    print(cluster_centers)

    # Generate and display the cluster summary table
    summary_df = generate_cluster_summary(clustered_data)
    print("\nCluster Summary Table:")
    print(summary_df)

    # List members of each cluster
    list_cluster_members(clustered_data)

    # Generate and display the cluster summary table
    summary_df = generate_cluster_summary(clustered_data)
    print("\nCluster Summary Table:")
    print(summary_df)

    # Visualize the attribute distribution for each cluster in separate figures
    visualize_clusters_separately(summary_df)

if __name__ == "__main__":
    main()  
