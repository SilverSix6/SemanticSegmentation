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
    
    # assign values outside of the range for the first iteration of old centers to ensure a second loop
    old_center1 = [-1,-1,-1]
    old_center2 = [-1,-1,-1]
    old_center3 = [-1,-1,-1]
    # contintue until convergence
    not_converged = True
    while(not_converged):
        # calculate the new average lab color of each cluster
        center1_lab = center(center1,slic_clusters)
        center2_lab = center(center2,slic_clusters)
        center3_lab = center(center3,slic_clusters)           
        # Step 2 compare the slic cluster average color to find the closest difference with the centers
        center1.clear()
        center2.clear()
        center3.clear()
        for i in range(size):
            diff1 = ((slic_clusters[i].l - center1_lab[0])**2 + (slic_clusters[i].a - center1_lab[1])**2 + (slic_clusters[i].b - center1_lab[2])**2)**0.5
            diff2 = ((slic_clusters[i].l - center2_lab[0])**2 + (slic_clusters[i].a - center2_lab[1])**2 + (slic_clusters[i].b - center2_lab[2])**2)**0.5
            diff3 = ((slic_clusters[i].l - center3_lab[0])**2 + (slic_clusters[i].a - center3_lab[1])**2 + (slic_clusters[i].b - center3_lab[2])**2)**0.5
            if diff1 < diff2 and diff1 < diff3:
                center1.append(slic_clusters[i].cid)
            elif diff2 < diff1 and diff2 < diff3:
                center2.append(slic_clusters[i].cid)
            else:
                center3.append(slic_clusters[i].cid)
        # Step 3 check for convergence
        if center1_lab == old_center1 and center2_lab == old_center2 and center3_lab == old_center3:
            not_converged = False
        else:
            old_center1 = center1_lab
            old_center2 = center2_lab
            old_center3 = center3_lab
    # Return the mega clusters
    return [center1,center2,center3]
# calculates center of the cluster's average lab value
def center(cluster,slic_clusters):
    for i in range(len(cluster)):
        sum_l += slic_clusters[cluster[i]].l
        sum_a += slic_clusters[cluster[i]].a
        sum_b += slic_clusters[cluster[i]].b
    return [sum_l/len(cluster),sum_a/len(cluster),sum_b/len(cluster)]
    