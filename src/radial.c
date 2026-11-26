#include "radial.h"

#include <ctype.h>
#include <math.h>
#include <string.h>

/* The learnable score vector is a linear spline of radius.
   Rotating the input by +120 degrees cyclically permutes the three scores. */
static const double ridge = 1e-5;
static const double sin120 = 0.86602540378443864676;

void radial_init(Radial *m, const Point *train, int n)
{
    *m = (Radial){0};
    for (int i = 0; i < n; ++i)
        m->radius = fmax(m->radius, hypot(train[i].x[0], train[i].x[1]));
    if (m->radius == 0.0)
        m->radius = 1.0;
}

/* At most two adjacent knots have nonzero support. Features for class k
   use the coordinates of R^(-k) x, with R the +120 degree rotation. */
static int features(const Radial *m, const double x[2], double f[3][4])
{
    double t = fmin(hypot(x[0], x[1]) / m->radius, 1.0) * (RADIAL_KNOTS - 1);
    int knot = (int)t;
    if (knot == RADIAL_KNOTS - 1)
        --knot;
    double mix = t - knot;
    double u[3] = {x[0], -0.5*x[0] + sin120*x[1], -0.5*x[0] - sin120*x[1]};
    double v[3] = {x[1], -sin120*x[0] - 0.5*x[1], sin120*x[0] - 0.5*x[1]};
    for (int k = 0; k < 3; ++k) {
        f[k][0] = (1.0 - mix) * u[k];
        f[k][1] = (1.0 - mix) * v[k];
        f[k][2] = mix * u[k];
        f[k][3] = mix * v[k];
    }
    return 2*knot;
}

static double score(const double *weights, double f[3][4], double prob[3], int label)
{
    double z[3] = {0};
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 4; ++j)
            z[k] += weights[j] * f[k][j];
    double peak = fmax(z[0], fmax(z[1], z[2])), total = 0.0;
    for (int k = 0; k < 3; ++k) {
        prob[k] = exp(z[k] - peak);
        total += prob[k];
    }
    if (!isfinite(total))
        return NAN;
    for (int k = 0; k < 3; ++k)
        prob[k] /= total;
    return label < 0 ? 0.0 : (peak - z[label]) + log(total);
}

double radial_predict(const Radial *m, const double x[2], double prob[3], int label)
{
    if (!isfinite(x[0]) || !isfinite(x[1]) || label > 2 || !(m->radius > 0.0))
        return NAN;
    double f[3][4];
    int start = features(m, x, f);
    return score(m->w + start, f, prob, label);
}

/* The Hessian is the softmax-weighted covariance of the class features
   plus ridge * I. Compute it only when the caller requests a Newton step. */
static double objective(const Radial *m, const Point *points, int n, double *g,
                        double h[][RADIAL_PARAMS])
{
    double loss = 0.0;
    if (g) {
        memset(g, 0, RADIAL_PARAMS * sizeof(*g));
        memset(h, 0, RADIAL_PARAMS * sizeof(*h));
    }
    for (int i = 0; i < n; ++i) {
        double f[3][4], p[3];
        int a = features(m, points[i].x, f), y = points[i].label;
        loss += score(m->w + a, f, p, y);
        if (!g)
            continue;
        double mean[4] = {0};
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k)
                mean[j] += p[k] * f[k][j];
            g[a+j] += (mean[j] - f[y][j]) / n;
        }
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 4; ++j)
                for (int l = 0; l <= j; ++l)
                    h[a+j][a+l] += p[k] * (f[k][j] - mean[j]) * (f[k][l] - mean[l]) / n;
    }
    loss /= n;
    for (int i = 0; i < RADIAL_PARAMS; ++i) {
        loss += 0.5 * ridge * m->w[i] * m->w[i];
        if (g) {
            g[i] += ridge * m->w[i];
            h[i][i] += ridge;
        }
    }
    return loss;
}

static int cholesky_solve(double h[][RADIAL_PARAMS], const double g[], double delta[])
{
    for (int i = 0; i < RADIAL_PARAMS; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = h[i][j];
            for (int k = 0; k < j; ++k)
                value -= h[i][k] * h[j][k];
            if (i == j && !(value > 0.0))
                return 0;
            h[i][j] = i == j ? sqrt(value) : value / h[j][j];
        }
        double rhs = g[i];
        for (int j = 0; j < i; ++j)
            rhs -= h[i][j] * delta[j];
        delta[i] = rhs / h[i][i];
    }
    for (int i = RADIAL_PARAMS - 1; i >= 0; --i) {
        for (int j = i + 1; j < RADIAL_PARAMS; ++j)
            delta[i] -= h[j][i] * delta[j];
        delta[i] /= h[i][i];
    }
    return 1;
}

int radial_step(Radial *m, const Point *train, int n)
{
    double g[RADIAL_PARAMS], h[RADIAL_PARAMS][RADIAL_PARAMS], delta[RADIAL_PARAMS];
    double before = objective(m, train, n, g, h), decrement = 0.0;
    if (!isfinite(before) || !cholesky_solve(h, g, delta))
        return -1;
    for (int i = 0; i < RADIAL_PARAMS; ++i)
        decrement += g[i] * delta[i];
    if (!isfinite(decrement) || decrement < 0)
        return -1;
    if (decrement <= 1e-12)
        return 0;
    for (double step = 1.0; step >= 1e-8; step *= 0.5) {
        Radial candidate = *m;
        for (int i = 0; i < RADIAL_PARAMS; ++i)
            candidate.w[i] -= step * delta[i];
        double after = objective(&candidate, train, n, NULL, NULL);
        if (isfinite(after) && after <= before - 1e-4 * step * decrement) {
            *m = candidate;
            return 1;
        }
    }
    return -1;
}

int radial_save(const Radial *m, FILE *fp)
{
    if (fprintf(fp, "C3RADIAL1 %d\n%.17g\n", RADIAL_KNOTS, m->radius) < 0)
        return 0;
    for (int i = 0; i < RADIAL_PARAMS; ++i)
        if (fprintf(fp, "%.17g\n", m->w[i]) < 0)
            return 0;
    return 1;
}

int radial_load(Radial *m, const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;
    Radial candidate = {0};
    char magic[16], knots[16], expected[16];
    snprintf(expected, sizeof(expected), "%d", RADIAL_KNOTS);
    int ok = fscanf(fp, "%15s %15s %lf", magic, knots, &candidate.radius) == 3 &&
             strcmp(magic, "C3RADIAL1") == 0 && strcmp(knots, expected) == 0 &&
             candidate.radius > 0.0 && isfinite(candidate.radius);
    for (int i = 0; ok && i < RADIAL_PARAMS; ++i)
        ok = fscanf(fp, "%lf", &candidate.w[i]) == 1 && isfinite(candidate.w[i]);
    int c;
    while (ok && (c = fgetc(fp)) != EOF)
        ok = isspace((unsigned char)c);
    ok = ok && !ferror(fp);
    if (fclose(fp) || !ok)
        return 0;
    *m = candidate;
    return 1;
}

int radial_check(void)
{
    const Point data[] = {{{0.15, -0.2}, 0}, {{0.62, 0.4}, 2}, {{-0.2, 0.09}, 1}};
    Radial m;
    radial_init(&m, data, 3);
    for (int j = 0; j < RADIAL_PARAMS; ++j)
        m.w[j] = sin(j + 0.3);
    double grad[RADIAL_PARAMS], h[RADIAL_PARAMS][RADIAL_PARAMS];
    double before = objective(&m, data, 3, grad, h);
    for (int j = 0; j < RADIAL_PARAMS; ++j) {
        Radial plus = m, minus = m;
        plus.w[j] += 1e-5;
        minus.w[j] -= 1e-5;
        double g1[RADIAL_PARAMS], g2[RADIAL_PARAMS], scratch[RADIAL_PARAMS][RADIAL_PARAMS];
        double f1 = objective(&plus, data, 3, g1, scratch);
        double f2 = objective(&minus, data, 3, g2, scratch);
        if (fabs((f1 - f2) / 2e-5 - grad[j]) > 1e-8)
            return 0;
        for (int k = j; k < RADIAL_PARAMS; ++k)
            if (fabs((g1[k] - g2[k]) / 2e-5 - h[k][j]) > 1e-8)
                return 0;
    }
    if (radial_step(&m, data, 3) != 1 || objective(&m, data, 3, NULL, NULL) >= before)
        return 0;
    for (int i = 0; i < 3; ++i) {
        const double *x = data[i].x;
        double rx[2] = {-0.5*x[0] - sin120*x[1], sin120*x[0] - 0.5*x[1]}, p[3], q[3];
        radial_predict(&m, x, p, -1);
        radial_predict(&m, rx, q, -1);
        for (int k = 0; k < 3; ++k)
            if (fabs(p[k] - q[(k + 1) % 3]) > 1e-12)
                return 0;
    }
    puts("Radialnetz: Gradient, Hesse-Matrix, Newton-Abstieg und Rotationsrelation bestanden.");
    return 1;
}
