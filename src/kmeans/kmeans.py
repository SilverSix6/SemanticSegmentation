# Takes in the clusters of superpixels from the slic algorithm 
# performs k-means clustering based on the cluster's average lab color
# returns the mega clusters
def kmeans(clusters):
    print("K-means clustering")
    # Step 1 Randomly assign 3 or 4 centers at a random l a b
    
    # Step 2 compare the slic cluster average color to find the closest difference with the centers
    # Step 3 move centers to average lab of newly assigned clust
    # Repeat until convergence
    # Return the mega clusters