# Takes in the clusters of superpixels from the slic algorithm 
# performs k-means clustering based on the cluster's average lab color
# returns the mega clusters
def kmeans(slic_clusters):
    # Step 1 Randomly assign 3 or 4 centers at a random l a b
    center1 = []
    center2 = []
    center3 = []
    size = len(slic_clusters)
    for i in range(size):
        if i % 3 == 0:
            center1.append(slic_clusters[i].cid)
        elif i % 3 == 1:
            center2.append(slic_clusters[i].cid)
        else:
            center3.append(slic_clusters[i].cid)
    # calculate the average lab color of each cluster
    center1_lab = center(center1,slic_clusters)
    center2_lab = center(center2,slic_clusters)
    center3_lab = center(center3,slic_clusters)           
    # Step 2 compare the slic cluster average color to find the closest difference with the centers
    center1.clear()
    center2.clear()
    center3.clear()
        
    # Step 3 move centers to average lab of newly assigned clust
    # Repeat until convergence
    # Return the mega clusters
  
# calculates center of the cluster's average lab value
def center(cluster,slic_clusters):
    for i in range(len(cluster)):
        sum_l += slic_clusters[cluster[i]].l
        sum_a += slic_clusters[cluster[i]].a
        sum_b += slic_clusters[cluster[i]].b
    return [sum_l/len(cluster),sum_a/len(cluster),sum_b/len(cluster)]
    