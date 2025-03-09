import math

import cv2

class ClusterCenter:
    x = -1
    y = -1
    l = 0
    a = 0
    b = 0

def slic(image, m, s):
    # Initialization
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    cluster_centers = []

    for row in lab_image.shape[0]:
        for col in lab_image.shape[1]:
            



def distance_function(image, cluster_center: ClusterCenter, px, py, m):
    s = 10 # Maximum spatial distance expected within a given cluster
    l, a, b = image[px, py]

    d_c = math.sqrt((cluster_center.l - l) * (cluster_center.l - l)
                    + (cluster_center.a - a) * (cluster_center.a - a)
                    + (cluster_center.b - b) * (cluster_center.b - b))
    d_s = math.sqrt((cluster_center.x - px) * (cluster_center.x - px)
                    + (cluster_center.y - py) * (cluster_center.y - py))

    return math.sqrt(d_c * d_c + (d_s / s) * (d_s / s) * m * m)

