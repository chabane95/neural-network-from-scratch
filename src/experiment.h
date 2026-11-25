#ifndef SPIRAL_EXPERIMENT_H
#define SPIRAL_EXPERIMENT_H

/* The output directory must already exist. Returns 1 on success. */
int run_experiment(const char *out, unsigned seed, int epochs, int width, double lr);

#endif
