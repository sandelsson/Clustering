import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import cophenet



MAX_CLUSTERS = 10
MIN_CLUSTERS = 2
SHEETUNUMBER = 10
DISTANCE = "euclidean"
#DISTANCE = "rogerstanimoto"

OPTIMAL_K = 6

def data_preprocess():
    # Load the Excel file
    data = pd.read_excel('Freemium list.xlsx', sheet_name='Sheet' + str(SHEETUNUMBER))

    # Set 'Service Name' as the index
    data.set_index('Service Name', inplace=True)

    # Convert all feature columns to numeric values, replacing errors with NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill any NaN values with 0 (assuming missing values indicate absence)
    data = data.fillna(0)

    # Convert all non-zero values to 1 (binary conversion)
    #data = data.astype(bool).astype(int)
    
    # Step 5: Duplicate rows
    duplicate_rows = data[data.duplicated()]
    print(f"Number of duplicate rows: {len(duplicate_rows)}")
    
    if len(duplicate_rows) > 0:
        print("Duplicate rows:")
        print(duplicate_rows)
        data = data.drop_duplicates()
    
    
    return data
def dice_distance(data):
    # Compute the condensed distance matrix using the Dice distance
    condensed_distance = pdist(data.values, metric=DISTANCE)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(
        linkage_matrix,
        labels=data.index,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Service Name')
    plt.ylabel('Distance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return linkage_matrix, condensed_distance
def elbow_chart(linkage_matrix):
    
    last_merges = linkage_matrix[-(MAX_CLUSTERS - 1):]

    # Extract the distances at which clusters are merged
    distances = last_merges[:, 2]

    # Reverse the distances to go from upper left to lower right
    distances_reversed = distances[::-1]

    # Calculate the number of clusters corresponding to each merge
    num_clusters = np.arange(2, MAX_CLUSTERS + 1)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(num_clusters, distances_reversed, marker='o')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Linkage Distance')
    plt.xticks(num_clusters)
    plt.grid(True)
    plt.show()

    return None
def silhouette(condensed_distance):
    # Convert to a square distance matrix
    distance_matrix = squareform(condensed_distance)

    # Perform hierarchical clustering using the linkage function
    linkage_matrix = linkage(condensed_distance, method='average')

    # Define the range of clusters to evaluate
    range_n_clusters = list(range(MIN_CLUSTERS, MAX_CLUSTERS))

    # List to store silhouette scores
    silhouette_avg_scores = []

    for n_clusters in range_n_clusters:
        # Obtain cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Compute the silhouette score
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        silhouette_avg_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg}")

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters (Dice Distance)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(range_n_clusters)
    plt.grid(True)
    plt.show()
def ch_index(data, linkage_matrix):
    # Define the range of clusters to evaluate
    range_n_clusters = list(range(MIN_CLUSTERS, MAX_CLUSTERS))

    # List to store Calinski-Harabasz Index scores
    calinski_harabasz_scores = []

    for n_clusters in range_n_clusters:
        # Obtain cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Compute the Calinski-Harabasz Index
        ch_score = calinski_harabasz_score(data.values, cluster_labels)
        calinski_harabasz_scores.append(ch_score)
        print(f"For n_clusters = {n_clusters}, the Calinski-Harabasz Index is: {ch_score}")

    # Plot the Calinski-Harabasz Index
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, calinski_harabasz_scores, marker='o', color='orange')
    plt.title('Calinski-Harabasz Index for Different Numbers of Clusters (Dice Distance)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Index')
    plt.xticks(range_n_clusters)
    plt.grid(True)
    plt.show()
def gap(data, linkage_matrix):
    B = 10  # Number of reference datasets
    k_range = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)
    Wks = []

    # Compute within-cluster dispersion for the real data
    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        Wk = compute_Wk(squareform(pdist(data.values, metric='euclidean')), labels)
        Wks.append(Wk)

    Wks_ref = np.zeros((len(k_range), B))
    n_samples, n_features = data.shape
    feature_probs = data.mean(axis=0).values

    for b in range(B):
        # Generate random reference data with the same marginal distributions
        random_data = (np.random.rand(n_samples, n_features) < feature_probs).astype(int)

        # Check if there are NaN values in the random data
        if np.any(np.isnan(random_data)):
            raise ValueError(f"Random data contains NaN values in iteration {b}.")
        
        # Compute the condensed distance matrix for the random data
        random_distance_matrix = pdist(random_data, metric='euclidean')

        # Check if there are any non-finite values in the distance matrix
        if not np.all(np.isfinite(random_distance_matrix)):
            raise ValueError(f"Random distance matrix contains non-finite values in iteration {b}.")

        # Perform hierarchical clustering on reference data
        random_linkage_matrix = linkage(random_distance_matrix, method='ward')

        for idx, k in enumerate(k_range):
            random_labels = fcluster(random_linkage_matrix, k, criterion='maxclust')
            Wks_ref[idx, b] = compute_Wk(squareform(random_distance_matrix), random_labels)

    logWks = np.log(Wks)
    mean_logWks_ref = np.mean(np.log(Wks_ref), axis=1)
    gap_values = mean_logWks_ref - logWks

    # Plot the Gap Statistic
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, gap_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic for Determining Optimal Number of Clusters')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # Print the optimal number of clusters
    #optimal_k = next((k for i, k in enumerate(k_range[:-1]) if gap_values[i] >= gap_values[i + 1]), k_range[-1])
    optimal_k = k_range[np.argmax(gap_values)]
    print(f"Optimal number of clusters according to Gap Statistic: {optimal_k}")
def compute_Wk(distance_matrix, labels):
    Wk = 0.0
    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) > 1:
            cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            Wk += np.sum(cluster_distances) / 2.0  # Divide by 2 to avoid double counting
    return Wk  
def phi(data):
    #data.set_index('Service Name', inplace=True)

    feature_columns = data.columns.drop('Cluster', errors='ignore')



    # Initialize an empty DataFrame to store phi coefficients
    phi_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)

    # Calculate the phi coefficient for each pair of features
    for col1 in feature_columns:
        for col2 in feature_columns:
            if col1 == col2:
                phi_matrix.loc[col1, col2] = 1.0  # The phi coefficient with itself is 1
            else:
                # Create a contingency table
                contingency_table = pd.crosstab(data[col1], data[col2])
                # Compute the chi-squared statistic
                chi2, p, dof, expected = chi2_contingency(contingency_table, correction=False)
                # Compute phi coefficient
                n = contingency_table.values.sum()
                phi = np.sqrt(chi2 / n)
                # Assign the value to the matrix
                phi_matrix.loc[col1, col2] = phi

    # Convert the phi_matrix to numeric values
    phi_matrix = phi_matrix.astype(float)

    # Display the phi coefficient matrix
    print(phi_matrix)

    # Optional: Visualize the phi coefficient matrix as a heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(phi_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Phi Coefficient Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
def cluster_seeds(data, linkage_matrix, condensed_distance):
    
    # Step 5: Identify initial medoids for each cluster
    distance_matrix = squareform(condensed_distance)  # Convert to a square distance matrix
    initial_medoids = []

    cluster_labels = fcluster(linkage_matrix, OPTIMAL_K, criterion='maxclust')

    # Iterate through each cluster to find the medoid (point with minimum distance sum to others)
    for cluster_label in range(1, OPTIMAL_K + 1):
        # Find all points in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]

        if len(cluster_indices) == 0:
            continue

        # Extract submatrix of distances within the cluster
        sub_distance_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Calculate the total distance of each point to all others in the cluster
        total_distances = np.sum(sub_distance_matrix, axis=1)

        # Find the point with the minimum total distance as the medoid
        min_index = np.argmin(total_distances)
        medoid_index = cluster_indices[min_index]

        # Append the medoid index
        initial_medoids.append(medoid_index)

    # Print the initial medoids
    print("Initial medoids based on hierarchical clustering:")
    for i, medoid_index in enumerate(initial_medoids):
        print(f"Cluster {i + 1}: Medoid at index {medoid_index} ({data.index[medoid_index]})")

    return initial_medoids




def main():
    data = data_preprocess()

    #PCA

    #pca_df, pca = run_pca(data, n_components=2, scale_data=False)

    #phi-coefficient

    phi(data)

    #Dedrogram
    linkage_matrix, condensed_distance = dice_distance(data)

    #Cophenetic Correlation Coefficient
    coph_corr, coph_dists = cophenet(linkage_matrix, condensed_distance)
    print(f"Cophenetic Correlation Coefficient (CCC): {coph_corr:.4f}")

    #Elbow chart
    elbow_chart(linkage_matrix)

    #Silhouette score

    silhouette(condensed_distance)

    #Calinski-Harabasz Index

    ch_index(data, linkage_matrix)

    #Gap statistic

    gap(data, linkage_matrix)   

    #cluster seeds based on hierarchical clustering to be used in k-medoids clustering
    initial_medoids = cluster_seeds(data, linkage_matrix, condensed_distance)

    print(initial_medoids)
    #initial_medoids = [21,31,29,1,5,26] 



if __name__ == "__main__":
    main()