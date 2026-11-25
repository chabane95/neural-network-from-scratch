#ifndef NEURAL_NETWORK_NN_H
#define NEURAL_NETWORK_NN_H

#include <stddef.h>
#include <stdint.h>

#define NN_MAX_WIDTH 256

typedef struct {
    uint64_t state;
} RNG;

/* Stack-allocate as NN model = {0}; release with nn_free. Do not shallow-copy. */
typedef struct {
    int width;
    size_t parameter_count;
    uint64_t steps;
    double *storage;
    double *parameters;  /* Followed by Adam's first and second moments. */
    double *next_state;
    double *gradient;
} NN;

void rng_seed(RNG *rng, uint32_t seed);
double rng_uniform(RNG *rng);
int nn_init(NN *net, int width, RNG *rng);
void nn_free(NN *net);

/* Returns cross-entropy for label 0..2 or zero for prediction alone.
   Invalid inputs and numerical failures return NAN. */
double nn_predict(const NN *net, const double x[2], double probs[3], int label);

/* Returns mean batch loss before the update. NAN leaves the learned state intact. */
double nn_train_batch(NN *net, const double (*x)[2], const int *y, int count, double lr);

/* The portable inference checkpoint stores both hidden layers and the output.
   nn_load expects an empty model. It starts Adam afresh if training continues. */
int nn_save(const NN *net, const char *path);
int nn_load(NN *net, const char *path);
int nn_selftest(void);

#endif
