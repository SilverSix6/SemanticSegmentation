#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

/**
 * Enforces connectivity of pixels. Assigns pixels to be its most common neighboor. 
 * If there is a tie, the resulting label is randomly choosen between the tieing groups.
 * 
 * @param image: A segmented image.
 * @param width: The image's width (number of pixels)
 * @param height: The image's height (number of pixels) 
 */
void enforce_connectivity(int *image, int width, int height);

#endif