//
// Created by Chris Kjellqvist on 6/29/23.
//

#ifndef KMEANS_HACKING_CONSTANTS_H
#define KMEANS_HACKING_CONSTANTS_H

#define K 16
#define N_DATAS 50000

#ifdef __ARM64_ARCH_8__
#define HAS_FP16
using float_type = __fp16;
#else
using float_type = float;
#endif

#define MINIMUM_PERMISSIBLE_DATA_VALUE float_type(-10)

#endif //KMEANS_HACKING_CONSTANTS_H
