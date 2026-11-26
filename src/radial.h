#ifndef SPIRAL_RADIAL_H
#define SPIRAL_RADIAL_H

#include <stdio.h>

enum { RADIAL_KNOTS = 8, RADIAL_PARAMS = 2 * RADIAL_KNOTS };
typedef struct {
    double x[2];
    int label;
} Point;

typedef struct {
    double radius;
    double w[RADIAL_PARAMS];
} Radial;

void radial_init(Radial *model, const Point *train, int n);
double radial_predict(const Radial *model, const double x[2], double prob[3], int label);
/* One damped Newton update on the mean penalized training loss.
   Returns -1 on failure, 0 at convergence, and 1 after an accepted step. */
int radial_step(Radial *model, const Point *train, int n);
int radial_save(const Radial *model, FILE *out);
int radial_load(Radial *model, const char *path);
int radial_check(void);

#endif
