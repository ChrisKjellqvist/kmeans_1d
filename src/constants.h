//
// Created by Chris Kjellqvist on 6/29/23.
//

#ifndef KMEANS_HACKING_CONSTANTS_H
#define KMEANS_HACKING_CONSTANTS_H

#include "util.h"

// Due to error propagation in the k-means algorithm, the data values can
// stray from the real analytical result...
// We're using the smallest expressible value in float 16 as the epsilon

const double EPSILON = radix2float(1);

#define MINIMUM_PERMISSIBLE_DATA_VALUE float_type(-20)

#endif //KMEANS_HACKING_CONSTANTS_H
