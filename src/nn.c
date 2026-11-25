#include "nn.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* SplitMix64. The top 53 bits map exactly onto the binary64 unit interval. */
void rng_seed(RNG *rng, uint32_t seed)
{
    rng->state = seed;
}

double rng_uniform(RNG *rng)
{
    uint64_t z = (rng->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

/* Each output neuron owns a contiguous row of input weights. */
typedef struct {
    size_t b1, w2, b2, w3, b3, total;
} Layout;

static Layout layout(int width)
{
    size_t h = (size_t)width;
    Layout l;
    l.b1 = 2 * h;
    l.w2 = l.b1 + h;
    l.b2 = l.w2 + h * h;
    l.w3 = l.b2 + h;
    l.b3 = l.w3 + 3 * h;
    l.total = l.b3 + 3;
    return l;
}

static int allocate_model(NN *net, int width)
{
    if (!net || net->storage || width < 1 || width > NN_MAX_WIDTH)
        return 0;
    size_t n = layout(width).total;
    double *storage = calloc(7 * n, sizeof(*storage));
    if (!storage)
        return 0;
    net->width = width;
    net->parameter_count = n;
    net->steps = 0;
    net->storage = storage;
    net->parameters = storage;
    net->next_state = storage + 3 * n;
    net->gradient = storage + 6 * n;
    return 1;
}

static void xavier(double *weights, int inputs, int outputs, RNG *rng)
{
    double limit = sqrt(6.0 / (inputs + outputs));
    for (int i = 0; i < inputs * outputs; ++i)
        weights[i] = (2.0 * rng_uniform(rng) - 1.0) * limit;
}

int nn_init(NN *net, int width, RNG *rng)
{
    if (!rng || !allocate_model(net, width))
        return 0;
    Layout l = layout(width);
    xavier(net->parameters, 2, width, rng);
    xavier(net->parameters + l.w2, width, width, rng);
    xavier(net->parameters + l.w3, width, 3, rng);
    return 1;
}

void nn_free(NN *net)
{
    if (net) {
        free(net->storage);
        *net = (NN){0};
    }
}

static double dot(const double *a, const double *b, int count)
{
    double sum = 0.0;
    for (int i = 0; i < count; ++i)
        sum += a[i] * b[i];
    return sum;
}

typedef struct {
    double a1[NN_MAX_WIDTH], a2[NN_MAX_WIDTH], prob[3];
} Forward;

static double forward(const NN *net, Layout l, const double x[2], int label, Forward *f)
{
    const double *p = net->parameters;
    int h = net->width;
    for (int i = 0; i < h; ++i)
        f->a1[i] = tanh(p[l.b1 + i] + p[2*i] * x[0] + p[2*i+1] * x[1]);
    for (int i = 0; i < h; ++i)
        f->a2[i] = tanh(p[l.b2 + i] + dot(p + l.w2 + i*h, f->a1, h));

    double logits[3];
    for (int k = 0; k < 3; ++k)
        logits[k] = p[l.b3 + k] + dot(p + l.w3 + k*h, f->a2, h);
    double maximum = fmax(logits[0], fmax(logits[1], logits[2]));
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
        f->prob[k] = exp(logits[k] - maximum);
        sum += f->prob[k];
    }
    if (!isfinite(sum) || sum <= 0.0)
        return NAN;
    for (int k = 0; k < 3; ++k)
        f->prob[k] /= sum;
    if (label < 0 || label >= 3)
        return 0.0;
    /* Subtract logits first. Adding log(sum) to a large maximum loses precision. */
    return (maximum - logits[label]) + log(sum);
}

double nn_predict(const NN *net, const double x[2], double probs[3], int label)
{
    if (!net || !net->storage || !x || !probs || !isfinite(x[0]) || !isfinite(x[1]))
        return NAN;
    Forward f;
    double loss = forward(net, layout(net->width), x, label, &f);
    memcpy(probs, f.prob, sizeof(f.prob));
    return loss;
}

static void backward(NN *net, Layout l, const double x[2], int label, const Forward *f)
{
    double *g = net->gradient;
    const double *p = net->parameters;
    int h = net->width;
    double d2[NN_MAX_WIDTH], d1[NN_MAX_WIDTH];
    memset(d2, 0, (size_t)h * sizeof(*d2));
    memset(d1, 0, (size_t)h * sizeof(*d1));

    for (int k = 0; k < 3; ++k) {
        double d = f->prob[k] - (k == label);
        g[l.b3 + k] += d;
        size_t row = l.w3 + k*h;
        for (int j = 0; j < h; ++j) {
            g[row + j] += d * f->a2[j];
            d2[j] += p[row + j] * d;
        }
    }
    for (int i = 0; i < h; ++i) {
        double d = d2[i] * (1.0 - f->a2[i] * f->a2[i]);
        g[l.b2 + i] += d;
        size_t row = l.w2 + i*h;
        for (int j = 0; j < h; ++j) {
            g[row + j] += d * f->a1[j];
            d1[j] += p[row + j] * d;
        }
    }
    for (int i = 0; i < h; ++i) {
        double d = d1[i] * (1.0 - f->a1[i] * f->a1[i]);
        g[l.b1 + i] += d;
        g[2*i] += d * x[0];
        g[2*i+1] += d * x[1];
    }
}

static double batch_gradient(NN *net, const double (*x)[2], const int *y, int count)
{
    Layout l = layout(net->width);
    memset(net->gradient, 0, l.total * sizeof(double));
    Forward f;
    double loss = 0.0;
    for (int i = 0; i < count; ++i) {
        if (y[i] < 0 || y[i] >= 3 || !isfinite(x[i][0]) || !isfinite(x[i][1]))
            return NAN;
        double sample_loss = forward(net, l, x[i], y[i], &f);
        if (!isfinite(sample_loss))
            return NAN;
        loss += sample_loss;
        backward(net, l, x[i], y[i], &f);
    }
    return loss / count;
}

double nn_train_batch(NN *net, const double (*x)[2], const int *y, int count, double lr)
{
    if (!net || !net->storage || !x || !y || count < 1 || lr <= 0.0 ||
        !isfinite(lr) || net->steps == UINT64_MAX)
        return NAN;
    double loss = batch_gradient(net, x, y, count);
    if (!isfinite(loss))
        return NAN;

    size_t n = net->parameter_count;
    double *current = net->parameters, *next = net->next_state;
    double t = (double)(net->steps + 1);
    double correct1 = 1.0 / (1.0 - pow(0.9, t));
    double correct2 = 1.0 / (1.0 - pow(0.999, t));
    double scale = 1.0 / count;
    for (size_t i = 0; i < n; ++i) {
        double g = net->gradient[i] * scale;
        double m = 0.9 * current[n+i] + 0.1 * g;
        double v = 0.999 * current[2*n+i] + 0.001 * g * g;
        double update = (m * correct1) / (sqrt(v * correct2) + 1e-8);
        next[i] = current[i] - lr * update;
        if (!isfinite(next[i]) || !isfinite(v) || !isfinite(v * correct2))
            return NAN;
        next[n+i] = m;
        next[2*n+i] = v;
    }
    net->parameters = next;
    net->next_state = current;
    ++net->steps;
    return loss;
}

/* On disk: magic, little-endian uint32 width, binary64 weights/biases in
   layout order, then FNV-1a 64 over the width and parameters. */
static const unsigned char magic[8] = "NNMLP1\r\n";
static const uint64_t hash_offset = UINT64_C(14695981039346656037);

static int binary64_supported(void)
{
    return sizeof(double) == 8 && FLT_RADIX == 2 && DBL_MANT_DIG == 53 &&
           DBL_MAX_EXP == 1024;
}

static uint64_t hash_byte(uint64_t h, unsigned char byte)
{
    return (h ^ byte) * UINT64_C(1099511628211);
}

static int write_le(FILE *fp, uint64_t value, unsigned int size, uint64_t *hash)
{
    unsigned char bytes[8];
    for (unsigned int j = 0; j < size; ++j) {
        bytes[j] = (unsigned char)(value >> (8 * j));
        if (hash)
            *hash = hash_byte(*hash, bytes[j]);
    }
    return fwrite(bytes, 1, size, fp) == size;
}

static int read_le(FILE *fp, uint64_t *value, unsigned int size, uint64_t *hash)
{
    unsigned char bytes[8];
    if (fread(bytes, 1, size, fp) != size)
        return 0;
    *value = 0;
    for (unsigned int j = 0; j < size; ++j) {
        *value |= (uint64_t)bytes[j] << (8 * j);
        if (hash)
            *hash = hash_byte(*hash, bytes[j]);
    }
    return 1;
}

static int write_checkpoint(FILE *fp, const NN *net)
{
    uint64_t hash = hash_offset;
    if (fwrite(magic, 1, sizeof(magic), fp) != sizeof(magic) ||
        !write_le(fp, (uint32_t)net->width, 4, &hash))
        return 0;
    for (size_t i = 0; i < net->parameter_count; ++i) {
        uint64_t bits;
        if (!isfinite(net->parameters[i]))
            return 0;
        memcpy(&bits, net->parameters + i, sizeof(bits));
        if (!write_le(fp, bits, 8, &hash))
            return 0;
    }
    return write_le(fp, hash, 8, NULL);
}

int nn_save(const NN *net, const char *path)
{
    if (!net || !net->storage || !path || !*path || !binary64_supported())
        return 0;
    size_t length = strlen(path);
    if (length > SIZE_MAX - 16)
        return 0;
    char *temporary = malloc(length + 16);
    if (!temporary)
        return 0;
    memcpy(temporary, path, length);
    FILE *fp = NULL;
    for (unsigned int attempt = 0; attempt < 64 && !fp; ++attempt) {
        snprintf(temporary + length, 16, ".tmp.%u", attempt);
        fp = fopen(temporary, "wbx");
    }
    int ok = 0;
    if (fp) {
        ok = write_checkpoint(fp, net);
        if (fclose(fp) != 0)
            ok = 0;
        if (ok) {
#ifdef _WIN32
            ok = MoveFileExA(temporary, path, MOVEFILE_REPLACE_EXISTING) != 0;
#else
            ok = rename(temporary, path) == 0;
#endif
        }
        if (!ok)
            remove(temporary);
    }
    free(temporary);
    return ok;
}

static int read_checkpoint(FILE *fp, NN *candidate)
{
    unsigned char header[sizeof(magic)];
    uint64_t hash = hash_offset, width, bits;
    if (fread(header, 1, sizeof(header), fp) != sizeof(header) ||
        memcmp(header, magic, sizeof(header)) != 0 ||
        !read_le(fp, &width, 4, &hash) || width < 1 || width > NN_MAX_WIDTH ||
        !allocate_model(candidate, (int)width))
        return 0;
    for (size_t i = 0; i < candidate->parameter_count; ++i) {
        if (!read_le(fp, &bits, 8, &hash))
            return 0;
        memcpy(candidate->parameters + i, &bits, sizeof(bits));
        if (!isfinite(candidate->parameters[i]))
            return 0;
    }
    return read_le(fp, &bits, 8, NULL) && bits == hash &&
           fgetc(fp) == EOF && !ferror(fp);
}

int nn_load(NN *net, const char *path)
{
    if (!net || net->storage || !path || !*path || !binary64_supported())
        return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return 0;
    NN candidate = {0};
    int ok = read_checkpoint(fp, &candidate);
    if (fclose(fp) != 0)
        ok = 0;
    if (ok)
        *net = candidate;
    else
        nn_free(&candidate);
    return ok;
}

static double test_loss(const NN *net, const double (*x)[2], const int *y, int count)
{
    double sum = 0.0, probs[3];
    for (int i = 0; i < count; ++i)
        sum += nn_predict(net, x[i], probs, y[i]);
    return sum / count;
}

static int check_gradients_and_adam(NN *net)
{
    static const double x[][2] = {{0.7, -0.1}, {-0.8, 0.9}, {0.2, 0.3}, {0.0, -0.9}};
    static const int y[] = {2, 0, 1, 2};
    const double eps = 1e-5, lr = 0.002;
    const int count = 4;
    double loss = batch_gradient(net, x, y, count);
    if (!isfinite(loss)) {
        fprintf(stderr, "Gradiententest: Der mittlere Verlust ist nicht endlich.\n");
        return 0;
    }
    double maximum_error = 0.0;
    for (size_t i = 0; i < net->parameter_count; ++i) {
        double old = net->parameters[i];
        net->parameters[i] = old + eps;
        double plus = test_loss(net, x, y, count);
        net->parameters[i] = old - eps;
        double minus = test_loss(net, x, y, count);
        net->parameters[i] = old;
        double numerical = (plus - minus) / (2.0 * eps);
        double analytical = net->gradient[i] / count;
        double error = fabs(numerical - analytical);
        double tolerance = 1e-8 + 1e-6 * fmax(fabs(numerical), fabs(analytical));
        if (!isfinite(numerical) || !isfinite(analytical) || error > tolerance) {
            fprintf(stderr, "Gradiententest: Parameter %zu hat analytisch %.12g und numerisch %.12g.\n",
                    i, analytical, numerical);
            return 0;
        }
        maximum_error = fmax(maximum_error, error);
    }
    printf("Gradiententest bestanden: %zu Parameter, maximale absolute Abweichung %.3e.\n",
           net->parameter_count, maximum_error);
    double before = nn_train_batch(net, x, y, count, lr);
    if (!isfinite(before) || fabs(before - loss) > 1e-12 || net->steps != 1) {
        fprintf(stderr, "Adam-Test: Verlust oder Schrittzahl nach dem ersten Update falsch.\n");
        return 0;
    }
    /* The staging buffer still holds the complete state before step one. */
    for (size_t i = 0; i < net->parameter_count; ++i) {
        double g = net->gradient[i] / count;
        double expected = net->next_state[i] - lr * g / (fabs(g) + 1e-8);
        if (!isfinite(expected) || !isfinite(net->parameters[i]) ||
            fabs(expected - net->parameters[i]) > 1e-12) {
            fprintf(stderr, "Adam-Test: Abweichung beim ersten Update von Parameter %zu.\n", i);
            return 0;
        }
    }
    if (!(test_loss(net, x, y, count) < loss)) {
        fprintf(stderr, "Adam-Test: Der Verlust ist beim ersten Update nicht gesunken.\n");
        return 0;
    }
    puts("Adam-Test bestanden: Das erste Update entspricht der Referenzrechnung und senkt den Verlust.");
    return 1;
}

static int check_softmax(NN *net)
{
    const double x[2] = {0.0, 0.0};
    double probs[3];
    memset(net->parameters, 0, net->parameter_count * sizeof(double));
    double *out = net->parameters + layout(net->width).b3;
    out[0] = -1500.0;
    out[1] = out[2] = 1200.0;
    double loss = nn_predict(net, x, probs, 0);
    if (!isfinite(loss) || fabs(loss - (2700.0 + log(2.0))) > 1e-12 ||
        probs[0] != 0.0 || probs[1] != 0.5 || probs[2] != 0.5) {
        fprintf(stderr, "Softmax-Test: Falscher Verlust oder falsche Verteilung bei extremen Logits.\n");
        return 0;
    }
    out[0] = out[1] = out[2] = 1e30;
    loss = nn_predict(net, x, probs, 2);
    if (!isfinite(loss) || fabs(loss - log(3.0)) >= 1e-12) {
        fprintf(stderr, "Softmax-Test: Gleiche grosse Logits ergeben nicht den Verlust log(3).\n");
        return 0;
    }
    puts("Softmax-Test bestanden: Wahrscheinlichkeiten und Kreuzentropie bei extremen Logits.");
    return 1;
}

int nn_selftest(void)
{
    NN net = {0};
    RNG rng;
    rng_seed(&rng, 927);
    if (!nn_init(&net, 4, &rng)) {
        fprintf(stderr, "Selbsttest: Das Testmodell konnte nicht initialisiert werden.\n");
        return 0;
    }
    int ok = check_gradients_and_adam(&net) && check_softmax(&net);
    nn_free(&net);
    return ok;
}
