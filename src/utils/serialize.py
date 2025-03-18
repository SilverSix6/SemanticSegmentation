import pickle

def write_matrix(matrix, file):
    pickle.dump(matrix, file)

def read_matrix(file):
    return pickle.load(file)

def write_clusters(clusters, file):
    pickle.dump(clusters, file)

def read_clusters(file):
    return pickle.load(file)