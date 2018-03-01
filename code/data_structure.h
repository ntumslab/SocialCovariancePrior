#ifndef __DATA_STRUCTURE_H__
#define __DATA_STRUCTURE_H__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#define BUFFER_SIZE 4096

typedef struct DICT{
	int key;
	double value;
} Dict;

typedef struct BINARY_LIST{
	int rowCount;
	int *columnCounts;
	int **entry;
} BinaryList;

typedef struct LIST{
	int rowCount;
	int *columnCounts;
	Dict **entry;
} List;

typedef struct MATRIX{
	int rowCount;
	int columnCount;
	double **entry;
} Matrix;

typedef struct ARRAY_ENTRY{
	int key1;
	int key2;
	double value;
} ArrayEntry;

typedef struct ARRAY{
	int length;
	ArrayEntry *entry;
} Array;

typedef struct OPTIMIZER{
	int epochCount;
	double convergenceThreshold;
	double decay;
	double conditionConstant;
} Optimizer;

//==============================================================================================

double randomSampleStardardUniformVariable();
double randomSampleUniformVariable(double lowerBound, double upperBound);
int randomSampleInteger(int lowerBound, int upperBound);
double randomSampleStandardNormalVariable();
double randomSampleNormalVariable(double mean, double standardDeviation);

//==============================================================================================

int dictCompareAscendingKeys(const void *a, const void *b);
double dictVectorCalculateValueMean(Dict *vector, int length);
// The two vectors should be sorted by keys
double dictVectorCalculatePearsonCorrelationCoefficient(Dict* vector1, Dict* vector2, int length1, int length2, int intersectionLowerBound);
double dictVectorCalculateJaccardCoefficient(Dict *vector1, Dict *vector2, int length1, int length2);

//==============================================================================================

double vectorCalculateEuclideanDistanceSquare(double *vector1, double *vector2, int length);
double vectorCalculateCorrelationCoefficient(double *vector1, double *vector2, int length);
double vectorCalculateDotProduct(double *vector1, double *vector2, int length);
double vectorCalculateMean(double *vector, int length);
void vectorRunElimination(double *vector1, double *vector2, int length, int factorIndex);

//==============================================================================================

void listInitialize(List *list, int rowCount);
void listReleaseSpace(List *list);
void listAddEntry(List *list, int row, int key, double value);
void listCopy(List *source, List *target);
void listSortRows(List *list);
void listNormalizeRows(List *list);
int listCountEntries(List *list);
// columnVectors should contain at least the number of columns that list contains
void listGetReverseList(List *list, List *reverseList);
void listPrint(List *list, FILE *outputStream);
void listScan(List *list, FILE *inputStream);
void listCopyFromArray(List *list, Array *array, bool key1_2);

//==============================================================================================

void matrixInitialize(Matrix *matrix, int rowCount, int columnCount);
void matrixReleaseSpace(Matrix *matrix);
void matrixSetRandomValues(Matrix *matrix, double minValue, double maxValue);
void matrixSetNormallyRandomValues(Matrix *matrix, double mean, double standardDeviation);
// Source and target should contain the same rows and columns
void matrixCopyEntries(Matrix *source, Matrix *target);
void matrixPrint(Matrix *matrix, FILE *outputStream);
void matrixScan(Matrix *matrix, FILE *inputStream);
void matrixSetIdentity(Matrix *matrix);
void matrixSetValue(Matrix *matrix, double value);
void matrixMultiplyScalar(Matrix *matrix, double scalar);
void matrixAddScalar(Matrix *matrix, double scalar);
double matrixCalculateSquareSum(Matrix *matrix);
void matrixGetTranspose(Matrix *matrix, Matrix *transpose);
bool matrixGetInverse(Matrix *matrix, Matrix *inverse);
double matrixCalculatePositiveDefiniteLogDeterminant(Matrix *matrix);

//==============================================================================================

void arrayInitialize(Array *array);
void arrayReleaseSpace(Array *array);
void arrayAddEntry(Array* array, int key1, int key2, double value);
void arrayCopyFromList(Array *array, List *list);
void arrayShuffle(Array *array);
void arrayScan(Array *array, FILE *inputStream);
void arrayPrint(Array *array, FILE *outputStream);
void arraySort(Array *array, int (*compare)(const void *p, const void *q));

//==============================================================================================

void ratingReadHeader(char *ratingFilePath, int *rowCount, int *columnCount);
void ratingReadArray(char *ratingFilePath, Array *ratingArray);
void ratingReadList(char *ratingFilePath, List *ratingList);
void ratingReadMatrix(char *ratingFilePath, Matrix *ratingMatrix);
void ratingWriteMatrix(char *ratingFilePath, Matrix *ratingMatrix);
void ratingWriteArray(char *ratingFilePath, Array *ratingArray, int key1Count, int key2Count);
void ratingWriteList(char *ratingFilePath, List *ratingList, int columnCount);
void ratingNormalizeArrayByMean(Array *ratingArray);
void ratingNormalizeListByMean(List *ratingList);
void ratingSplitArrayValidation(Array *ratingArray, Array *trainingRatingArray, Array* validationRatingArray, float validationRatio);
void ratingSplitListValidation(List *ratingList, List *trainingRatingList, List* validationRatingList, float validationRatio);

//==============================================================================================

void optimizerInitialize(Optimizer *optimizer);
double optimizerAdadeltaGetStep(Optimizer *optimizer, double gradient, double *squaredStep, double *secondMoment);

#endif
