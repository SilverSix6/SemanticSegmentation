from utils.serialize import read_clusters, read_matrix
from kmeans.kmeans import kmeans_cluster

def run_single_kmeans():
    
    try:
        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")
        
        # Load the matrix and cluster
        matrix = read_matrix(matrix_file)
        cluster = read_clusters(cluster_file)

        # Do kmeans on super pixels
        kmeans_cluster(matrix,cluster)
        print("kmeans ran")
        
    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")