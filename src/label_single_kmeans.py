from src.utils.serialize import read_clusters, read_matrix

def run_single_kmeans():
    
    try:
        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")
        
        # Load the matrix and cluster
        matrix = read_matrix(matrix_file)
        cluster = read_clusters(cluster_file)
        
        # Do kmeans on super pixels
        
    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")