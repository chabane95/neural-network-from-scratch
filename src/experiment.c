#include "experiment.h"
#include "nn.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    CLASSES = 3,
    TRAIN_N = CLASSES * 250,
    VAL_N = CLASSES * 70,
    TEST_N = CLASSES * 100,
    BATCH = 32,
    GRID = 61,
    SNAPSHOT_INTERVALS = 16
};

static const double tau = 6.28318530717958647693;

typedef struct {
    double x[2];
    int label;
} Point;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

typedef struct {
    double weights[CLASSES][3]; /* Features (x1, x2, 1), then softmax across classes. */
} LinearModel;

static char *output_path(const char *dir, const char *name)
{
    size_t length = strlen(dir) + strlen(name) + 2;
    char *path = malloc(length);
    if (path)
        snprintf(path, length, "%s/%s", dir, name);
    return path;
}

static FILE *open_output(const char *dir, const char *name)
{
    char *path = output_path(dir, name);
    FILE *fp = path ? fopen(path, "w") : NULL;
    if (!fp)
        fprintf(stderr, "Datei kann nicht angelegt werden: %s/%s\n", dir, name);
    free(path);
    return fp;
}

static int close_output(FILE *fp)
{
    int ok = !ferror(fp);
    if (fclose(fp) != 0)
        ok = 0;
    if (!ok)
        fputs("Fehler beim Schreiben der Ergebnisdateien.\n", stderr);
    return ok;
}

/* Box-Muller yields two independent standard normal variates. */
static void normal_pair(RNG *rng, double *a, double *b)
{
    double magnitude = sqrt(-2.0 * log(1.0 - rng_uniform(rng)));
    double phase = tau * rng_uniform(rng);
    *a = magnitude * cos(phase);
    *b = magnitude * sin(phase);
}

static void generate_split(Point *points, int count, uint32_t seed)
{
    RNG rng;
    rng_seed(&rng, seed);
    for (int i = 0; i < count; ++i) {
        double radial_fraction = rng_uniform(&rng);
        double radius = 0.04 + 0.96 * radial_fraction;
        int label = i % CLASSES;
        double angular_noise, unused, dx, dy;
        normal_pair(&rng, &angular_noise, &unused);
        normal_pair(&rng, &dx, &dy);
        double angle = tau * label / CLASSES + 4.8 * radial_fraction + 0.28 * angular_noise;
        points[i].x[0] = radius * cos(angle) + 0.035 * dx;
        points[i].x[1] = radius * sin(angle) + 0.035 * dy;
        points[i].label = label;
    }
}

static void write_point_rows(FILE *fp, const Point *points, int count, const char *split)
{
    for (int i = 0; i < count; ++i)
        fprintf(fp, "%.17g,%.17g,%d,%s\n", points[i].x[0], points[i].x[1],
                points[i].label, split);
}

static int write_points(const char *dir, const Point train[], const Point val[], const Point test[])
{
    FILE *fp = open_output(dir, "points.csv");
    if (!fp)
        return 0;
    fputs("x1,x2,label,split\n", fp);
    write_point_rows(fp, train, TRAIN_N, "train");
    write_point_rows(fp, val, VAL_N, "val");
    write_point_rows(fp, test, TEST_N, "test");
    return close_output(fp);
}

static int evaluate(const NN *net, const Point *points, int count, Metrics *m, FILE *predictions)
{
    int correct = 0;
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        const Point *p = &points[i];
        double prob[CLASSES];
        double loss = nn_predict(net, p->x, prob, p->label);
        if (!isfinite(loss))
            return 0;
        total += loss;
        int best = 0;
        for (int k = 0; k < CLASSES; ++k) {
            if (!isfinite(prob[k]))
                return 0;
            if (prob[k] > prob[best])
                best = k;
        }
        correct += best == p->label;
        if (predictions)
            fprintf(predictions, "%.17g,%.17g,%d,%d,%.17g,%.17g,%.17g\n",
                    p->x[0], p->x[1], p->label, best, prob[0], prob[1], prob[2]);
    }
    m->loss = total / count;
    m->accuracy = (double)correct / count;
    return isfinite(m->loss) && (!predictions || !ferror(predictions));
}

static double linear_predict(const LinearModel *model, const Point *point, double prob[CLASSES])
{
    double logits[CLASSES];
    for (int k = 0; k < CLASSES; ++k) {
        const double *w = model->weights[k];
        logits[k] = w[0] * point->x[0] + w[1] * point->x[1] + w[2];
    }
    const double maximum = fmax(logits[0], fmax(logits[1], logits[2]));
    double sum = 0.0;
    for (int k = 0; k < CLASSES; ++k) {
        prob[k] = exp(logits[k] - maximum);
        sum += prob[k];
    }
    for (int k = 0; k < CLASSES; ++k)
        prob[k] /= sum;
    return (maximum - logits[point->label]) + log(sum);
}

static int fit_linear(LinearModel *model, const Point train[])
{
    /* The softmax covariance has spectral norm <= 1/2. The mean Hessian
       therefore has norm <= L = mean(||(x1,x2,1)||^2)/2. */
    double squared_features = 0.0;
    for (int i = 0; i < TRAIN_N; ++i) {
        const double *x = train[i].x;
        squared_features += x[0] * x[0] + x[1] * x[1] + 1.0;
    }
    const double step = 2.0 * TRAIN_N / squared_features;
    for (int iteration = 0; iteration < 5000; ++iteration) {
        LinearModel gradient = {0};
        for (int i = 0; i < TRAIN_N; ++i) {
            double prob[CLASSES];
            linear_predict(model, &train[i], prob);
            for (int k = 0; k < CLASSES; ++k) {
                double delta = prob[k] - (train[i].label == k);
                gradient.weights[k][0] += delta * train[i].x[0];
                gradient.weights[k][1] += delta * train[i].x[1];
                gradient.weights[k][2] += delta;
            }
        }
        double norm_squared = 0.0;
        for (int k = 0; k < CLASSES; ++k) {
            for (int j = 0; j < 3; ++j) {
                double g = gradient.weights[k][j] / TRAIN_N;
                norm_squared += g * g;
                model->weights[k][j] -= step * g;
            }
        }
        if (!isfinite(norm_squared))
            return 0;
        if (norm_squared <= 1e-16)
            return 1;
    }
    fputs("Lineares Vergleichsmodell erreicht die Gradiententoleranz nicht.\n", stderr);
    return 0;
}

static Metrics evaluate_linear(const LinearModel *model, const Point test[])
{
    double loss = 0.0;
    int correct = 0;
    for (int i = 0; i < TEST_N; ++i) {
        double p[CLASSES];
        loss += linear_predict(model, &test[i], p);
        int best = 0;
        for (int k = 1; k < CLASSES; ++k)
            if (p[k] > p[best])
                best = k;
        correct += best == test[i].label;
    }
    return (Metrics){loss / TEST_N, (double)correct / TEST_N};
}

static int write_surface(FILE *fp, const NN *net, int frame, int epoch)
{
    const double extent = 1.15;
    for (int row = 0; row < GRID; ++row) {
        double x2 = -extent + 2 * extent * row / (GRID - 1);
        for (int col = 0; col < GRID; ++col) {
            double x[2] = {-extent + 2 * extent * col / (GRID - 1), x2};
            double p[CLASSES];
            double status = nn_predict(net, x, p, -1);
            if (!isfinite(status) || !isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]))
                return 0;
            fprintf(fp, "%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g\n", frame, epoch,
                    x[0], x[1], p[0], p[1], p[2]);
        }
        if (ferror(fp))
            return 0;
    }
    return 1;
}

static int train_epoch(NN *net, const Point train[], int order[], RNG *shuffle, double lr)
{
    for (int i = TRAIN_N - 1; i > 0; --i) {
        int j = (int)(rng_uniform(shuffle) * (i + 1));
        int swap = order[i];
        order[i] = order[j];
        order[j] = swap;
    }
    for (int start = 0; start < TRAIN_N; start += BATCH) {
        int count = TRAIN_N - start < BATCH ? TRAIN_N - start : BATCH;
        double x[BATCH][2];
        int y[BATCH];
        for (int i = 0; i < count; ++i) {
            const Point *p = &train[order[start + i]];
            x[i][0] = p->x[0];
            x[i][1] = p->x[1];
            y[i] = p->label;
        }
        if (!isfinite(nn_train_batch(net, (const double (*)[2])x, y, count, lr)))
            return 0;
    }
    return 1;
}

/* Sample epochs on a quadratic schedule to resolve the early updates. */
static int is_snapshot_epoch(int epoch, int epochs)
{
    const int denominator = SNAPSHOT_INTERVALS * SNAPSHOT_INTERVALS;
    for (int i = 0; i <= SNAPSHOT_INTERVALS; ++i) {
        long long target = ((long long)epochs * i * i + denominator - 1) / denominator;
        if (target == epoch)
            return 1;
    }
    return 0;
}

static int write_summary(const char *dir, unsigned seed, int width, int epochs, int best_epoch,
                         double lr, Metrics test, Metrics linear)
{
    FILE *fp = open_output(dir, "summary.csv");
    if (!fp)
        return 0;
    fputs("seed,width,epochs,best_epoch,train_n,val_n,test_n,test_loss,test_accuracy,"
          "lr,batch_size,linear_test_loss,linear_test_accuracy\n", fp);
    fprintf(fp, "%u,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n",
            seed, width, epochs, best_epoch, TRAIN_N, VAL_N, TEST_N, test.loss, test.accuracy,
            lr, BATCH, linear.loss, linear.accuracy);
    return close_output(fp);
}

int run_experiment(const char *out, unsigned seed, int epochs, int width, double lr)
{
    if (!out || !*out || epochs < 1 || epochs == INT_MAX || width < 1 ||
        !isfinite(lr) || lr <= 0.0)
        return 0;

    Point train[TRAIN_N], val[VAL_N], test[TEST_N];
    generate_split(train, TRAIN_N, seed ^ UINT32_C(0xa34691c5));
    generate_split(val, VAL_N, seed ^ UINT32_C(0x62841f37));
    generate_split(test, TEST_N, seed ^ UINT32_C(0xd09b53e1));
    if (!write_points(out, train, val, test))
        return 0;

    RNG init, shuffle;
    rng_seed(&init, seed ^ UINT32_C(0x296db24f));
    rng_seed(&shuffle, seed ^ UINT32_C(0xc1543a09));
    int order[TRAIN_N];
    for (int i = 0; i < TRAIN_N; ++i)
        order[i] = i;

    NN model = {0}, selected = {0};
    char *checkpoint = output_path(out, "model.nn");
    FILE *history = open_output(out, "history.csv");
    FILE *surface = open_output(out, "surface.csv");
    FILE *predictions = NULL;
    int ok = 0, best_epoch = 0, frame = 0;
    double best_loss = HUGE_VAL;
    Metrics test_metrics = {0}, linear_metrics = {0};
    LinearModel linear = {0};
    if (!checkpoint || !history || !surface || !nn_init(&model, width, &init))
        goto cleanup;
    fputs("epoch,train_loss,val_loss,train_accuracy,val_accuracy\n", history);
    fputs("frame,epoch,x1,x2,p0,p1,p2\n", surface);

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        Metrics tr, va;
        if (epoch > 0 && !train_epoch(&model, train, order, &shuffle, lr))
            goto cleanup;
        if (!evaluate(&model, train, TRAIN_N, &tr, NULL) ||
            !evaluate(&model, val, VAL_N, &va, NULL))
            goto cleanup;
        if (fprintf(history, "%d,%.17g,%.17g,%.17g,%.17g\n", epoch, tr.loss, va.loss,
                    tr.accuracy, va.accuracy) < 0)
            goto cleanup;
        if (va.loss < best_loss) {
            if (!nn_save(&model, checkpoint))
                goto cleanup;
            best_loss = va.loss;
            best_epoch = epoch;
        }
        if (is_snapshot_epoch(epoch, epochs) && !write_surface(surface, &model, frame++, epoch))
            goto cleanup;
    }

    if (!nn_load(&selected, checkpoint) || !write_surface(surface, &selected, frame, best_epoch) ||
        !fit_linear(&linear, train))
        goto cleanup;
    predictions = open_output(out, "predictions.csv");
    if (!predictions)
        goto cleanup;
    fputs("x1,x2,label,pred,p0,p1,p2\n", predictions);
    linear_metrics = evaluate_linear(&linear, test);
    ok = isfinite(linear_metrics.loss) && evaluate(&selected, test, TEST_N, &test_metrics, predictions);

cleanup:
    if (history && !close_output(history))
        ok = 0;
    if (surface && !close_output(surface))
        ok = 0;
    if (predictions && !close_output(predictions))
        ok = 0;
    free(checkpoint);
    nn_free(&selected);
    nn_free(&model);
    if (!ok) {
        fputs("Experiment fehlgeschlagen (Rechnung, Modellspeicher oder Dateiausgabe).\n", stderr);
        return 0;
    }
    if (!write_summary(out, seed, width, epochs, best_epoch, lr, test_metrics, linear_metrics))
        return 0;
    printf("Ausgewaehlte Epoche %d/%d | Validierungsverlust %.5f | Testverlust %.5f | "
           "Testgenauigkeit %.2f %% (%d Beispiele)\n", best_epoch, epochs, best_loss,
           test_metrics.loss, 100 * test_metrics.accuracy, TEST_N);
    printf("Lineares Vergleichsmodell | Testverlust %.5f | Testgenauigkeit %.2f %%\n",
           linear_metrics.loss, 100 * linear_metrics.accuracy);
    return 1;
}
