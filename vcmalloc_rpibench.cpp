#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>

extern "C" {
#include "vcmalloc.h"
}

#pragma warning(disable:4996)

using namespace std;

void log_csv(
    const char* filename,
    const char* operation,
    const char* x,
    const char* y,
    const char* z,
    const char* xtype,
    const char* ytype,
    const char* ztype,
    const char* xunit,
    const char* yunit,
    const char* zunit
) {
    printf(
        "%s\n"
        "%s (%s): %s\n"
        "%s (%s): %s\n"
        "%s (%s): %s\n",
        operation,
        ztype, zunit, z,
        xtype, xunit, x,
        ytype, yunit, y
    );

    FILE* csv_file = fopen(filename, "a");

    fseek(csv_file, 0, SEEK_END);
    if (ftell(csv_file) == 0) {
        fprintf(csv_file, "operation, x, y, z, xtype, ytype, ztype, xunit, yunit, zunit\n");
    }

    if (csv_file) {
        fprintf(csv_file, "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n",
            operation, x, y, z, xtype, ytype, ztype, xunit, yunit, zunit);
        fclose(csv_file);
    }
}


typedef struct {
    double* features;
    size_t label;
} knnPoint;

double euclideanDistance(const double* a, const double* b, size_t dimensions) {
    double distance = 0.0;
    for (size_t i = 0; i < dimensions; ++i) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

void kmeans(double** centroids, double** points, size_t* assignments, size_t* count, size_t k, size_t pointsCount, size_t dimensions, size_t Iter) {
    for (size_t iter = 0; iter < Iter; ++iter) {
        for (size_t i = 0; i < pointsCount; ++i) {
            double minDistance = euclideanDistance(points[i], centroids[0], dimensions);
            size_t closest = 0;
            for (size_t j = 1; j < k; ++j) {
                double distance = euclideanDistance(points[i], centroids[j], dimensions);
                if (distance < minDistance) {
                    minDistance = distance;
                    closest = j;
                }
            }
            assignments[i] = closest;
        }

        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                centroids[i][j] = 0.0;
            }
        }

        for (size_t i = 0; i < pointsCount; ++i) {
            size_t cluster = assignments[i];
            count[cluster]++;
            for (size_t j = 0; j < dimensions; ++j) {
                centroids[cluster][j] += points[i][j];
            }
        }

        for (size_t i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (size_t j = 0; j < dimensions; ++j) {
                    centroids[i][j] /= count[i];
                }
            }
        }
        memset(count, 0, k * sizeof(size_t));
    }
}
void kmeans_affine(double** centroids, double** points, size_t* assignments, size_t* count, size_t k, size_t pointsCount, size_t dimensions, size_t Iter) {
    
    double* affine_centroids = centroids[0];
    double* affine_points = points[0];
    
    for (size_t iter = 0; iter < Iter; ++iter) {
        for (size_t i = 0; i < pointsCount; ++i) {
            double minDistance = euclideanDistance(points[i], centroids[0], dimensions);
            size_t closest = 0;
            for (size_t j = 1; j < k; ++j) {
                double distance = euclideanDistance(points[i], centroids[j], dimensions);
                if (distance < minDistance) {
                    minDistance = distance;
                    closest = j;
                }
            }
            assignments[i] = closest;
        }

        

        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                //centroids[i][j] = 0.0;
                affine_centroids[i * dimensions + j] = 0.0;
            }
        }

        for (size_t i = 0; i < pointsCount; ++i) {
            size_t cluster = assignments[i];
            count[cluster]++;
            for (size_t j = 0; j < dimensions; ++j) {
                //centroids[cluster][j] += points[i][j];
                affine_centroids[cluster * dimensions + j] += affine_points[i * dimensions + j];
            }
        }

        for (size_t i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (size_t j = 0; j < dimensions; ++j) {
                    //centroids[i][j] /= count[i];
                    affine_centroids[i * dimensions + j] /= count[i];
                }
            }
        }
        memset(count, 0, k * sizeof(size_t));
    }
}

void knn(knnPoint* dataset, size_t datasetSize, knnPoint testPoint, size_t k, size_t dimensions, size_t* nearestNeighbors, double* distances) {

    for (size_t i = 0; i < datasetSize; ++i) {
        distances[i] = euclideanDistance(testPoint.features, dataset[i].features, dimensions);
    }

    for (size_t i = 0; i < k; ++i) {
        double minDistance = HUGE_VAL;
        size_t minIndex = -1;

        for (size_t j = 0; j < datasetSize; ++j) {
            if (distances[j] < minDistance) {
                minDistance = distances[j];
                minIndex = j;
            }
        }

        nearestNeighbors[i] = dataset[minIndex].label;
        distances[minIndex] = HUGE_VAL; // Set to a large value to exclude from further consideration
    }
}

void matmult(double** A, double** B, double** C, size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
			C[i][j] = 0.0;
            for (size_t k = 0; k < n; ++k) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}
void matmult_affine(double** A, double** B, double** C, size_t m, size_t n, size_t p) {
	double* affine_A = A[0];
	double* affine_B = B[0];
	double* affine_C = C[0];
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
			//C[i][j] = 0.0;
			affine_C[i * p + j] = 0.0;
            for (size_t k = 0; k < n; ++k) {
				//C[i][j] += A[i][k] * B[k][j];
				affine_C[i * p + j] += affine_A[i * n + k] * affine_B[k * p + j];
			}
		}
	}
}



int knn_m(int argc, char* argv[]) {
    const char* allocator_name = "m";

    size_t datasetSize, dimensions, k;

    if (argc != 4) {
        printf("Usage: %s <dataset size> <dimensions> <k>\n", argv[0]);
        return 1;
    }

    datasetSize = atoi(argv[1]);
    dimensions = atoi(argv[2]);
    k = atoi(argv[3]);

    knnPoint* dataset = (knnPoint*)malloc(datasetSize * sizeof(knnPoint));
    for (size_t i = 0; i < datasetSize; ++i) {
        dataset[i].features = (double*)malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            dataset[i].features[j] = i * dimensions + j;
        dataset[i].label = i;
    }

    knnPoint testPoint;
    testPoint.features = (double*)malloc(dimensions * sizeof(double));
    for (size_t j = 0; j < dimensions; ++j)
        testPoint.features[j] = j;
    testPoint.label = -1;

    size_t* nearestNeighbors = (size_t*)malloc(k * sizeof(size_t));

    double* distances = (double*)malloc(datasetSize * sizeof(double));

    clock_t start = clock();
    knn(dataset, datasetSize, testPoint, k, dimensions, nearestNeighbors, distances);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_knn.csv",
        "knn",

        to_string(datasetSize).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,

        "dataset size",
        "time",
        "allocator",

        "",
        "seconds",
        ""

    );


    for (size_t i = 0; i < datasetSize; ++i) {
        free(dataset[i].features);
    }
    free(dataset);

    free(testPoint.features);
    free(nearestNeighbors);
    free(distances);

    return 0;
}
int knn_vcm(int argc, char* argv[]) {
    const char* allocator_name = "vcm";

    size_t datasetSize, dimensions, k;

    if (argc != 4) {
        printf("Usage: %s <dataset size> <dimensions> <k>\n", argv[0]);
        return 1;
    }

    datasetSize = atoi(argv[1]);
    dimensions = atoi(argv[2]);
    k = atoi(argv[3]);

    size_t total_size =
        datasetSize * sizeof(knnPoint) +
        datasetSize * dimensions * sizeof(double) +
        dimensions * sizeof(double) +
        k * sizeof(size_t) +
        datasetSize * sizeof(double);

    size_t total_allocations = 
        1 + 
        datasetSize +
        1 +
        1 +
        1;

    hca_init(total_size, total_allocations, 1);

    knnPoint* dataset = (knnPoint*)vca_malloc(datasetSize * sizeof(knnPoint));
    for (size_t i = 0; i < datasetSize; ++i) {
        dataset[i].features = (double*)vca_malloc(dimensions * sizeof(double));

        for (size_t j = 0; j < dimensions; ++j)
            dataset[i].features[j] = i * dimensions + j;
        dataset[i].label = i;
    
    }

    knnPoint testPoint;
    testPoint.features = (double*)vca_malloc(dimensions * sizeof(double));

    for (size_t j = 0; j < dimensions; ++j)
        testPoint.features[j] = j;
    testPoint.label = -1;

    size_t* nearestNeighbors = (size_t*)vca_malloc(k * sizeof(size_t));

    double* distances = (double*)vca_malloc(datasetSize * sizeof(double));

    clock_t start = clock();
    knn(dataset, datasetSize, testPoint, k, dimensions, nearestNeighbors, distances);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_knn.csv",
        "knn",
        to_string(datasetSize).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,
        "dataset size",
        "time",
        "allocator",
        "",
        "seconds",
        ""

    );

    hcm_clear(&hca_hcm);
    

    return 0;
}

int kmeans_m(int argc, char* argv[]) {
    const char* allocator_name = "m";

    size_t pointsCount, dimensions, k, maxIterations;

    if (argc == 5) {
        pointsCount = atoi(argv[1]);
        dimensions = atoi(argv[2]);
        k = atoi(argv[3]);
        maxIterations = atoi(argv[4]);
    }
    else {
        printf("Usage: %s <pointsCount> <dimensions> <k> <Iterations>\n", argv[0]);
        return 1;
    }

    double** points = (double**)malloc(pointsCount * sizeof(double*));
    for (size_t i = 0; i < pointsCount; ++i) {
        points[i] = (double*)malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            points[i][j] = i * dimensions + j;
    }

    double** centroids = (double**)malloc(k * sizeof(double*));
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = (double*)malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            centroids[i][j] = i * dimensions + j;
    }

    size_t* assignments = (size_t*)malloc(pointsCount * sizeof(size_t));
    for (size_t i = 0; i < pointsCount; ++i)
        assignments[i] = 0;

    size_t* count = (size_t*)malloc(k * sizeof(size_t));

    clock_t start = clock();
    kmeans(centroids, points, assignments, count, k, pointsCount, dimensions, maxIterations);
    clock_t end = clock();

    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_kmeans.csv",
        "kmeans",

        to_string(maxIterations).c_str(),
        to_string(elapsed).c_str(),
        allocator_name,

        "iterations",
        "time",
        "allocator",

        "",
        "seconds",
        ""
    );

    for (size_t i = 0; i < k; ++i) {
        free(centroids[i]);
    }
    free(centroids);

    for (size_t i = 0; i < pointsCount; ++i) {
        free(points[i]);
    }
    free(points);

    free(assignments);
    free(count);

    return 0;
}
int kmeans_vcm(int argc, char* argv[]) {
    const char* allocator_name = "vcm";

    size_t pointsCount, dimensions, k, maxIterations;

    if (argc == 5) {
        pointsCount = atoi(argv[1]);
        dimensions = atoi(argv[2]);
        k = atoi(argv[3]);
        maxIterations = atoi(argv[4]);
    }
    else {
        printf("Usage: %s <pointsCount> <dimensions> <k> <Iterations>\n", argv[0]);
        return 1;
    }

    size_t total_size =
        pointsCount * sizeof(double*) +
		pointsCount * dimensions * sizeof(double) +
        k * sizeof(double*) +
		k * dimensions * sizeof(double) +
		pointsCount * sizeof(size_t) +
		k * sizeof(size_t);

    size_t total_allocations =
        1 + 
        pointsCount +
        1 +
        k +
        1 +
        1;

    hca_init(total_size, total_allocations, 1);


    double** points = (double**)vca_malloc(pointsCount * sizeof(double*));
    for (size_t i = 0; i < pointsCount; ++i) {
        points[i] = (double*)vca_malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            points[i][j] = i * dimensions + j;
    }

    double** centroids = (double**)vca_malloc(k * sizeof(double*));
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = (double*)vca_malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            centroids[i][j] = i * dimensions + j;
    }

    size_t* assignments = (size_t*)vca_malloc(pointsCount * sizeof(size_t));
    for (size_t i = 0; i < pointsCount; ++i)
        assignments[i] = 0;

    size_t* count = (size_t*)vca_malloc(k * sizeof(size_t));

    clock_t start = clock();
    kmeans(centroids, points, assignments, count, k, pointsCount, dimensions, maxIterations);
    clock_t end = clock();

    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_kmeans.csv",
        "kmeans",

        to_string(maxIterations).c_str(),
        to_string(elapsed).c_str(),
        allocator_name,

        "iterations",
        "time",
        "allocator",

        "",
        "seconds",
        ""
    );

    hcm_clear(&hca_hcm);

    return 0;
}
int kmeans_vcma(int argc, char* argv[]) {
    const char* allocator_name = "vcma";

    size_t pointsCount, dimensions, k, maxIterations;

    if (argc == 5) {
        pointsCount = atoi(argv[1]);
        dimensions = atoi(argv[2]);
        k = atoi(argv[3]);
        maxIterations = atoi(argv[4]);
    }
    else {
        printf("Usage: %s <pointsCount> <dimensions> <k> <Iterations>\n", argv[0]);
        return 1;
    }

    size_t total_size =
        pointsCount * sizeof(double*) +
        pointsCount * dimensions * sizeof(double) +
        k * sizeof(double*) +
        k * dimensions * sizeof(double) +
        pointsCount * sizeof(size_t) +
        k * sizeof(size_t);

    size_t total_allocations =
        1 +
        pointsCount +
        1 +
        k +
        1 +
        1;

    hca_init(total_size, total_allocations, 1);


    double** points = (double**)vca_malloc(pointsCount * sizeof(double*));
    for (size_t i = 0; i < pointsCount; ++i) {
        points[i] = (double*)vca_malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            points[i][j] = i * dimensions + j;
    }

    double** centroids = (double**)vca_malloc(k * sizeof(double*));
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = (double*)vca_malloc(dimensions * sizeof(double));
        for (size_t j = 0; j < dimensions; ++j)
            centroids[i][j] = i * dimensions + j;
    }

    size_t* assignments = (size_t*)vca_malloc(pointsCount * sizeof(size_t));
    for (size_t i = 0; i < pointsCount; ++i)
        assignments[i] = 0;

    size_t* count = (size_t*)vca_malloc(k * sizeof(size_t));

    clock_t start = clock();
    kmeans_affine(centroids, points, assignments, count, k, pointsCount, dimensions, maxIterations);
    clock_t end = clock();

    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_kmeans.csv",
        "kmeans",

        to_string(maxIterations).c_str(),
        to_string(elapsed).c_str(),
        allocator_name,

        "iterations",
        "time",
        "allocator",

        "",
        "seconds",
        ""
    );

    hcm_clear(&hca_hcm);

    return 0;
}

int matmult_m(int argc, char* argv[]) {
    
	const char* allocator_name = "m";

	size_t m, n, p;

    if (argc != 4) {
		printf("Usage: %s <m> <n> <p>\n", argv[0]);
		return 1;
	}

	m = atoi(argv[1]);
	n = atoi(argv[2]);
	p = atoi(argv[3]);

	double** A = (double**)malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
		A[i] = (double*)malloc(n * sizeof(double));
	}

	double** B = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; ++i) {
		B[i] = (double*)malloc(p * sizeof(double));
		for (size_t j = 0; j < p; ++j)
			B[i][j] = i * p + j;
	}

	double** C = (double**)malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
		C[i] = (double*)malloc(p * sizeof(double));
	}

    clock_t start = clock();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j)
            A[i][j] = i * n + j;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j)
            B[i][j] = i * p + j;
    }
    
    for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < p; ++j)
			C[i][j] = 0.0;
	}
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_matmult.csv",
        "linear access",

        to_string(m).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,
        
        "matrix order",
        "time",
        "allocator",

        "NxN",
        "seconds",
        ""
    );

	start = clock();
	matmult(A, B, C, m, n, p);
	end = clock();
	time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
		"vcmalloc_rpibench_matmult.csv",
		"matmult",

		to_string(m).c_str(),
		to_string(time_spent).c_str(),
		allocator_name,

		"matrix order",
		"time",
		"allocator",

		"NxN",
		"seconds",
		""
	);

    for (size_t i = 0; i < m; ++i) {
		free(A[i]);
	}
	free(A);

    for (size_t i = 0; i < n; ++i) {
		free(B[i]);
	}
	free(B);

    for (size_t i = 0; i < m; ++i) {
		free(C[i]);
	}
	free(C);

	return 0;
}
int matmult_vcm(int argc, char* argv[]) {

    const char* allocator_name = "vcm";

    size_t m, n, p;

    if (argc != 4) {
        printf("Usage: %s <m> <n> <p>\n", argv[0]);
        return 1;
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);

    size_t total_size =
		m * sizeof(double*) +
		m * n * sizeof(double) +
		n * sizeof(double*) +
		n * p * sizeof(double) +
		m * sizeof(double*) +
		m * p * sizeof(double);

    size_t total_allocations =
		1 +
		m +
		1 +
		n +
		1 +
		m;

    hca_init(total_size, total_allocations, 1);

    double** A = (double**)vca_malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
        A[i] = (double*)vca_malloc(n * sizeof(double));
        for (size_t j = 0; j < n; ++j)
            A[i][j] = i * n + j;
    }

    double** B = (double**)vca_malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; ++i) {
        B[i] = (double*)vca_malloc(p * sizeof(double));
        for (size_t j = 0; j < p; ++j)
            B[i][j] = i * p + j;
    }

    double** C = (double**)vca_malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
        C[i] = (double*)vca_malloc(p * sizeof(double));
    }

    clock_t start = clock();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j)
            A[i][j] = i * n + j;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j)
            B[i][j] = i * p + j;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j)
            C[i][j] = 0.0;
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_matmult.csv",
        "linear access",

        to_string(m).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,

        "matrix order",
        "time",
        "allocator",

        "NxN",
        "seconds",
        ""
    );

    start = clock();
    matmult(A, B, C, m, n, p);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_matmult.csv",
        "matmult",

        to_string(m).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,

        "matrix order",
        "time",
        "allocator",

        "NxN",
        "seconds",
        ""
    );

    hcm_clear(&hca_hcm);

    return 0;
}
int matmult_vcma(int argc, char* argv[]) {

    const char* allocator_name = "vcma";

    size_t m, n, p;

    if (argc != 4) {
        printf("Usage: %s <m> <n> <p>\n", argv[0]);
        return 1;
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);

    size_t total_size =
        m * sizeof(double*) +
        m * n * sizeof(double) +
        n * sizeof(double*) +
        n * p * sizeof(double) +
        m * sizeof(double*) +
        m * p * sizeof(double);

    size_t total_allocations =
        1 +
        m +
        1 +
        n +
        1 +
        m;

    hca_init(total_size, total_allocations, 1);

    double** A = (double**)vca_malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
        A[i] = (double*)vca_malloc(n * sizeof(double));
        for (size_t j = 0; j < n; ++j)
            A[i][j] = i * n + j;
    }

    double** B = (double**)vca_malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; ++i) {
        B[i] = (double*)vca_malloc(p * sizeof(double));
        for (size_t j = 0; j < p; ++j)
            B[i][j] = i * p + j;
    }

    double** C = (double**)vca_malloc(m * sizeof(double*));
    for (size_t i = 0; i < m; ++i) {
        C[i] = (double*)vca_malloc(p * sizeof(double));
    }

    clock_t start = clock();

    double * affine_A = A[0];
    double * affine_B = B[0];
    double * affine_C = C[0];

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j)
            //A[i][j] = i * n + j;
            affine_A[i * n + j] = i * n + j;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j)
            //B[i][j] = i * p + j;
            affine_B[i * p + j] = i * p + j;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j)
            //C[i][j] = 0.0;
            affine_C[i * p + j] = 0.0;
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_matmult.csv",
        "linear access",

        to_string(m).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,

        "matrix order",
        "time",
        "allocator",

        "NxN",
        "seconds",
        ""
    );

    start = clock();
    matmult_affine(A, B, C, m, n, p);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    log_csv(
        "vcmalloc_rpibench_matmult.csv",
        "matmult",

        to_string(m).c_str(),
        to_string(time_spent).c_str(),
        allocator_name,

        "matrix order",
        "time",
        "allocator",

        "NxN",
        "seconds",
        ""
    );

    hcm_clear(&hca_hcm);

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
		printf("Usage: %s <operation> <args>\n", argv[0]);
		return 1;
	}

	string operation = argv[1];

    if (operation == "knn_m") {
		return knn_m(argc - 1, argv + 1);
	}
    else if (operation == "knn_vcm") {
		return knn_vcm(argc - 1, argv + 1);
	}
    else if (operation == "kmeans_m") {
		return kmeans_m(argc - 1, argv + 1);
	}
    else if (operation == "kmeans_vcm") {
		return kmeans_vcm(argc - 1, argv + 1);
	}
    else if (operation == "kmeans_vcma") {
		return kmeans_vcma(argc - 1, argv + 1);
	}
    else if (operation == "matmult_m") {
		return matmult_m(argc - 1, argv + 1);
	}
    else if (operation == "matmult_vcm") {
		return matmult_vcm(argc - 1, argv + 1);
	}
    else if (operation == "matmult_vcma") {
		return matmult_vcma(argc - 1, argv + 1);
	}
    else {
		printf("Unknown operation: %s\n", operation.c_str());
		return 1;
	}
}   