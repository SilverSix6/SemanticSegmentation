from src.utils.segmentation_utils import load_matrix, load_cluster

def run_single_kmeans():
    
    try:
        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")
        
        # Load the matrix and cluster
        matrix = load_matrix(matrix_file)
        cluster = load_cluster(cluster_file)
        
        
        
    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")