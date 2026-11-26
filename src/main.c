#include "experiment.h"
#include "nn.h"
#include "radial.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define new_directory(path) _mkdir(path)
#else
#include <sys/stat.h>
#define new_directory(path) mkdir(path, 0777)
#endif

static void usage(FILE *stream)
{
    fputs("Neural network from scratch | 3-Spiralen-Klassifikation\n"
          "  spiralnet train [--out lauf] [--seed 42] [--epochs 200]\n"
          "                  [--width 32] [--lr 0.003]\n"
          "  spiralnet study [--out studie] [--seed 42] [--epochs 200] [--width 32] [--lr 0.003]\n"
          "  spiralnet predict MODELL X1 X2\n"
          "  spiralnet check\n"
          "\n"
          "train erzeugt einen neuen Ausgabeordner. Sein Elternordner muss existieren.\n"
          "study vergleicht fuenf aufeinanderfolgende Seeds ab dem angegebenen Startwert.\n"
          "width, epochs und lr konfigurieren das MLP; die Radialbasis besitzt acht Stuetzstellen.\n"
          "width setzt die Breite beider verdeckten MLP-Schichten (1 bis 256).\n"
          "predict schreibt die vorhergesagte Klasse und drei Wahrscheinlichkeiten als CSV.\n",
          stream);
}

static int unsigned_integer(const char *text, unsigned long limit, unsigned long *value)
{
    char *end;
    if (*text < '0' || *text > '9')
        return 0;
    errno = 0;
    *value = strtoul(text, &end, 10);
    return !errno && !*end && *value <= limit;
}

static int real_number(const char *text, double *value)
{
    char *end;
    errno = 0;
    *value = strtod(text, &end);
    return end != text && !*end && errno == 0 && isfinite(*value);
}

static int train(int argc, char **argv)
{
    unsigned seed = 42;
    int epochs = 200, width = 32;
    double lr = 0.003;
    int study = strcmp(argv[1], "study") == 0;
    const char *out = study ? "studie" : "lauf";

    for (int i = 2; i < argc; i += 2) {
        const char *option = argv[i];
        unsigned long number;
        double rate;
        if (i + 1 >= argc) {
            fprintf(stderr, "Wert fehlt fuer %s\n", option);
            return EXIT_FAILURE;
        }
        const char *arg = argv[i + 1];
        if (strcmp(option, "--out") == 0 && *arg) {
            out = arg;
        } else if (strcmp(option, "--seed") == 0 && unsigned_integer(arg, UINT32_MAX, &number)) {
            seed = (unsigned)number;
        } else if (strcmp(option, "--epochs") == 0 &&
                   unsigned_integer(arg, INT_MAX - 1, &number) && number > 0) {
            epochs = (int)number;
        } else if (strcmp(option, "--width") == 0 &&
                   unsigned_integer(arg, NN_MAX_WIDTH, &number) && number > 0) {
            width = (int)number;
        } else if (strcmp(option, "--lr") == 0 && real_number(arg, &rate) && rate > 0.0) {
            lr = rate;
        } else {
            fprintf(stderr, "Unbekannte Option oder ungueltiger Wert: %s %s\n", option, arg);
            return EXIT_FAILURE;
        }
    }

    if (study && seed > UINT32_MAX - 4) {
        fputs("Fuenf aufeinanderfolgende 32-Bit-Seeds muessen darstellbar sein.\n", stderr);
        return EXIT_FAILURE;
    }
    if (new_directory(out) != 0) {
        if (errno == EEXIST)
            fprintf(stderr, "Ordner oder Datei existiert bereits. Bitte mit --out einen neuen Lauf benennen.\n");
        else
            perror(out);
        return EXIT_FAILURE;
    }
    if (!(study ? run_study(out, seed, epochs, width, lr) : run_experiment(out, seed, epochs, width, lr))) {
        fprintf(stderr, "Training fehlgeschlagen. Unvollstaendige Ausgabe in %s\n", out);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int predict(const char *path, const char *first, const char *second)
{
    NN model = {0};
    double point[2], probabilities[3];
    if (!real_number(first, &point[0]) || !real_number(second, &point[1])) {
        fputs("X1 und X2 muessen endliche Zahlen sein. Dezimaltrennzeichen ist der Punkt.\n", stderr);
        return EXIT_FAILURE;
    }
    Radial radial;
    double status;
    if (nn_load(&model, path)) {
        status = nn_predict(&model, point, probabilities, -1);
        nn_free(&model);
    } else if (radial_load(&radial, path)) {
        status = radial_predict(&radial, point, probabilities, -1);
    } else {
        fprintf(stderr, "Modell konnte nicht geladen werden: %s\n", path);
        return EXIT_FAILURE;
    }
    if (!isfinite(status)) {
        fputs("Vorhersage fuer diese Eingabe numerisch nicht darstellbar.\n", stderr);
        return EXIT_FAILURE;
    }
    int best = 0;
    for (int k = 0; k < 3; ++k) {
        if (probabilities[k] > probabilities[best])
            best = k;
    }
    printf("x1,x2,pred,p0,p1,p2\n%.17g,%.17g,%d,%.17g,%.17g,%.17g\n",
           point[0], point[1], best, probabilities[0], probabilities[1], probabilities[2]);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        usage(stdout);
        return EXIT_SUCCESS;
    }
    if (argc >= 2 && (strcmp(argv[1], "train") == 0 || strcmp(argv[1], "study") == 0))
        return train(argc, argv);
    if (argc == 5 && strcmp(argv[1], "predict") == 0)
        return predict(argv[2], argv[3], argv[4]);
    if (argc == 2 && strcmp(argv[1], "check") == 0)
        return nn_selftest() && radial_check() ? EXIT_SUCCESS : EXIT_FAILURE;
    usage(stderr);
    return EXIT_FAILURE;
}
