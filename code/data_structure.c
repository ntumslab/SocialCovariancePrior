#include "data_structure.h"

double randomSampleStardardUniformVariable(){
	return (double)rand() / RAND_MAX;
}

double randomSampleUniformVariable(double lowerBound, double upperBound){
	if(lowerBound > upperBound){
		double temp = lowerBound;
		lowerBound = upperBound;
		upperBound = temp;
	}
	return lowerBound + randomSampleStardardUniformVariable() * (upperBound - lowerBound);
}

int randomSampleInteger(int lowerBound, int upperBound){
	if(lowerBound > upperBound){
		int temp = lowerBound;
		lowerBound = upperBound;
		upperBound = temp;
	}
	double uniform;
	do{
		uniform = randomSampleStardardUniformVariable();
	}while(uniform == 1.0);
	return (int)(lowerBound + uniform * (upperBound - lowerBound + 1));
}

double randomSampleStandardNormalVariable(){
	double sum = 0;
	for(int i = 0; i < 12; i ++){
		sum += randomSampleStardardUniformVariable();
	}
	return sum - 6;
}

double randomSampleNormalVariable(double mean, double standardDeviation){
	return mean + standardDeviation * randomSampleStandardNormalVariable();
}

//==============================================================================================

int dictCompareAscendingKeys(const void *a, const void *b){
	Dict p = *(Dict*)a;
	Dict q = *(Dict*)b;

	if(p.key < q.key){
		return -1;
	}
	if(p.key > q.key){
		return 1;
	}
	return 0;
}

double dictVectorCalculateValueMean(Dict *vector, int length){
	double mean = 0.0;
	for(int i = 0; i < length; i ++){
		mean += vector[i].value;
	}
	mean /= length;
	return mean;
}

// The two vectors should be sorted by keys
double dictVectorCalculatePearsonCorrelationCoefficient(Dict* vector1, Dict* vector2, int length1, int length2, int intersectionLowerBound){
	double covariance = 0.0;
	double norm1 = 0.0;
	double norm2 = 0.0;
	int intersectionCount = 0;

	double mean1 = dictVectorCalculateValueMean(vector1, length1);
	double mean2 = dictVectorCalculateValueMean(vector2, length2);

	for(int a = 0, b = 0; a < length1 && b < length2; ){
		if(vector1[a].key == vector2[b].key){
			double error1 = vector1[a].value - mean1;
			double error2 = vector2[b].value - mean2;

			covariance += error1 * error2;
			norm1 += error1 * error1;
			norm2 += error2 * error2;

			a += 1;
			b += 1;
			intersectionCount += 1;
		}
		else if(vector1[a].key < vector2[b].key){
			a += 1;
		}
		else{
			b += 1;
		}
	}

	return (intersectionCount > 0 && norm1 > 0.0 && norm2 > 0.0 && intersectionCount >= intersectionLowerBound)
		? covariance / (sqrt(norm1) * sqrt(norm2)) : 0.0;
}

// The two vectors should be sorted by keys
double dictVectorCalculateJaccardCoefficient(Dict *vector1, Dict *vector2, int length1, int length2){
	int intersectionCount = 0;

	for(int a = 0, b = 0; a < length1 && b < length2; ){
		if(vector1[a].key == vector2[b].key){
			intersectionCount += 1;
			a += 1;
			b += 1;
		}
		else if(vector1[a].key < vector2[b].key){
			a += 1;
		}
		else{
			b += 1;
		}
	}

	int unionCount = length1 + length2 - intersectionCount;

	return (unionCount > 0)? (double)intersectionCount / unionCount: 0.0;	
}

//==============================================================================================

double vectorCalculateDotProduct(double *vector1, double *vector2, int length){
	double dotProduct = 0.0;
	for(int d = 0; d < length; d ++){
		dotProduct += vector1[d] * vector2[d];
	}
	return dotProduct;
}

double vectorCalculateMean(double *vector, int length){
	double mean = 0.0;
	for(int d = 0; d < length; d ++){
		mean += vector[d];
	}
	mean /= length;
	return mean;
}

double vectorCalculateCovariance(double *vector1, double *vector2, int length, double mean1, double mean2){
	double covariance = 0.0;
	for(int d = 0; d < length; d ++){
		double bias1 = vector1[d] - mean1;
		double bias2 = vector2[d] - mean2;

		covariance += bias1 * bias2;
	}
	covariance /= length;
	return covariance;
}

double vectorCalculateCorrelationCoefficient(double *vector1, double *vector2, int length){
	double mean1 = vectorCalculateMean(vector1, length);
	double mean2 = vectorCalculateMean(vector2, length);
	double covariance = vectorCalculateCovariance(vector1, vector2, length, mean1, mean2);
	double norm1 = vectorCalculateCovariance(vector1, vector1, length, mean1, mean1);
	double norm2 = vectorCalculateCovariance(vector2, vector2, length, mean2, mean2);
	double correlationCoefficient = (norm1 > 0 && norm2 > 0) ? covariance / (sqrt(norm1) * sqrt(norm2)) : 0.0;	
	return correlationCoefficient;
}

double vectorCalculateEuclideanDistanceSquare(double *vector1, double *vector2, int length){
	double distance = 0.0;
	for(int d = 0; d < length; d ++){
		double difference = vector1[d] - vector2[d];
		distance += difference * difference;
	}
	return distance;
}

// Algorithm: Euclidean algorithm
void vectorRunElimination(double *vector1, double *vector2, int length, int factorIndex){
	if(abs(vector1[factorIndex]) < abs(vector2[factorIndex])){
		for(int d = 0; d < length; d ++){
			double temp = vector1[d];
			vector1[d] = vector2[d];
			vector2[d] = temp;
		}
	}

	if(vector2[factorIndex] != 0){
		double factor = vector1[factorIndex] / vector2[factorIndex];
		for(int d = 0; d < length; d ++){
			vector1[d] -= vector2[d] * factor; 
		}
	
		for(int d = 0; d < length; d ++){
			double temp = vector1[d];
			vector1[d] = vector2[d];
			vector2[d] = temp;
		}
	}
}

//==============================================================================================

void listInitialize(List *list, int rowCount){
	list -> rowCount = rowCount;
	list -> columnCounts = (int*)malloc(sizeof(int) * rowCount);
	list -> entry = (Dict**)malloc(sizeof(Dict*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		list -> columnCounts[row] = 0;
		list -> entry[row] = (Dict*)malloc(sizeof(Dict));
	}
}

void listReleaseSpace(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		free(list -> entry[row]);
	}
	free(list -> entry);
	free(list -> columnCounts);
	list -> entry = NULL;
	list -> columnCounts = NULL;
}

void listAddEntry(List *list, int row, int column, double value){
	int columnCount = list -> columnCounts[row];
	list -> entry[row] = (Dict*)realloc(list -> entry[row], sizeof(Dict) * (columnCount + 1));
	list -> entry[row][columnCount].key = column;
	list -> entry[row][columnCount].value = value;
	list -> columnCounts[row] += 1;
}

void listCopy(List *source, List *target){
	for(int row = 0; row < source -> rowCount; row ++){
		for(int j = 0; j < source -> columnCounts[row]; j ++){
			int column = source -> entry[row][j].key;
			double value = source -> entry[row][j].value;
			listAddEntry(target, row, column, value);
		}
	}
}

void listSortRows(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		qsort(list -> entry[row], list -> columnCounts[row], sizeof(Dict), dictCompareAscendingKeys);
	}
}

void listNormalizeRows(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		double valueSum = 0.0;

		for(int j = 0; j < list -> columnCounts[row]; j ++){
			valueSum += list -> entry[row][j].value;
		}
		
		if(valueSum != 0.0){
			for(int j = 0; j < list -> columnCounts[row]; j ++){
				list -> entry[row][j].value /= valueSum;
			}
		}
	}	
}

int listCountEntries(List *list){
	int entryCount = 0;
	for(int row = 0; row < list -> rowCount; row ++){
		entryCount += list -> columnCounts[row];
	}
	return entryCount;
}

// columnVectors should contain at least the number of columns that list contains
void listGetReverseList(List *list, List *reverseList){
	for(int row = 0; row < reverseList -> rowCount; row ++){
		reverseList -> columnCounts[row] = 0;
	}
	
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entry[row][j].key;
			double value = list -> entry[row][j].value;

			listAddEntry(reverseList, column, row, value);
		}
	}
}

void listPrint(List *list, FILE *outputStream){
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entry[row][j].key;
			double value = list -> entry[row][j].value;
			fprintf(outputStream, "%d\t%d\t%.10f\n", row, column, value);
		}
	}
}

void listScan(List *list, FILE *inputStream){
	int row, column;
	double value;
	char line[BUFFER_SIZE];

	while(fgets(line, BUFFER_SIZE, inputStream) != NULL){
		int readValueCount = sscanf(line, "%d%d%lf", &row, &column, &value);
		
		switch(readValueCount){
			case 3:
				listAddEntry(list, row, column, value);
				break;
			case 2:
				listAddEntry(list, row, column, 1);
				break;
		}
	}
}

void listCopyFromArray(List *list, Array *array, bool key1_2){
	for(int a = 0; a < array -> length; a ++){
		int key1 = array -> entry[a].key1;
		int key2 = array -> entry[a].key2;
		double value = array -> entry[a].value;

		if(key1_2){
			listAddEntry(list, key1, key2, value);
		}
		else{
			listAddEntry(list, key2, key1, value);
		}
	}
}

//==============================================================================================

void matrixInitialize(Matrix *matrix, int rowCount, int columnCount){
	matrix -> rowCount = rowCount;
	matrix -> columnCount = columnCount;
	matrix -> entry = (double**)malloc(sizeof(double*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		matrix -> entry[row] = (double*)malloc(sizeof(double) * columnCount);
	}	
}

void matrixReleaseSpace(Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		free(matrix -> entry[row]);
	}
	free(matrix -> entry);
	matrix -> entry = NULL;
}

void matrixSetRandomValues(Matrix *matrix, double minValue, double maxValue){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			double uniform = (double)rand() / RAND_MAX;
			matrix -> entry[row][column] = minValue + (maxValue - minValue) * uniform;
		}
	}
}

void matrixSetNormallyRandomValues(Matrix *matrix, double mean, double standardDeviation){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entry[row][column] = randomSampleNormalVariable(mean, standardDeviation);
		}
	}
}

void matrixCopyEntries(Matrix *source, Matrix *target){
	for(int row = 0; row < source -> rowCount; row ++){
		for(int column = 0; column < source -> columnCount; column ++){
			target -> entry[row][column] = source -> entry[row][column];
		}
	}
}

void matrixPrint(Matrix *matrix, FILE *outputStream){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			if(column > 0){
				fprintf(outputStream, "\t");
			}
			fprintf(outputStream, "%.10f", matrix -> entry[row][column]);
		}
		fprintf(outputStream, "\n");
	}
}

void matrixScan(Matrix *matrix, FILE *inputStream){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			int argumentCount = fscanf(inputStream, "%lf", &matrix -> entry[row][column]);
		}
	}	
}

void matrixSetIdentity(Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entry[row][column] = 0;
		}
		matrix -> entry[row][row] = 1;
	}
}

void matrixSetValue(Matrix *matrix, double value){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entry[row][column] = value;
		}
	}
}

void matrixMultiplyScalar(Matrix *matrix, double scalar){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entry[row][column] *= scalar;
		}
	}	
}

void matrixAddScalar(Matrix *matrix, double scalar){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entry[row][column] += scalar;
		}
	}	
}

double matrixCalculateSquareSum(Matrix *matrix){
	double squareSum = 0.0;

	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			double value = matrix -> entry[row][column];
			squareSum += value * value;
		}
	}

	return squareSum;
}

// The transpose should be of the same size (#rows and #columns) as the input matrix
void matrixGetTranspose(Matrix *matrix, Matrix *transpose){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			transpose -> entry[column][row] = matrix -> entry[row][column];
		}
	}
}

// Algorithm: Gaussian-Jordan elimination
// The inverse should be of the same size (#rows and #columns) as the input matrix
// Return value: Whether the inverse exists or not
bool matrixGetInverse(Matrix *matrix, Matrix *inverse){
	if(matrix -> rowCount != matrix -> columnCount){
		return false;
	}

	bool inverseExistent = true;
	matrixSetIdentity(inverse);

	Matrix copiedMatrix;
	matrixInitialize(&copiedMatrix, matrix -> rowCount, matrix -> columnCount);
	matrixCopyEntries(matrix, &copiedMatrix);

	for(int row = 0; row < copiedMatrix.rowCount; row ++){
		bool nonZeroRowFound = true;

		// Find a non-zero row
		if(copiedMatrix.entry[row][row] == 0){
			nonZeroRowFound = false;

			for(int anotherRow = row + 1; anotherRow < copiedMatrix.rowCount; anotherRow ++){
				if(copiedMatrix.entry[anotherRow][row] != 0){
					nonZeroRowFound = true;
					
					for(int column = row; column < copiedMatrix.columnCount; column ++){
						double temp1 = copiedMatrix.entry[row][column];
						copiedMatrix.entry[row][column] = copiedMatrix.entry[anotherRow][column];
						copiedMatrix.entry[anotherRow][column] = temp1;
					}
					for(int column = 0; column < inverse -> columnCount; column ++){
						double temp2 = inverse -> entry[row][column];
						inverse -> entry[row][column] = inverse -> entry[anotherRow][column];
						inverse -> entry[anotherRow][column] = temp2;
					}

					break;
				}
			}
		}

		if(nonZeroRowFound == false){
			inverseExistent = false;
			break;
		}
		
		// Run the elimination
		double factorReciprocal = 1.0 / copiedMatrix.entry[row][row];
		for(int column = row; column < copiedMatrix.columnCount; column ++){
			copiedMatrix.entry[row][column] *= factorReciprocal;
		}
		for(int column = 0; column < copiedMatrix.columnCount; column ++){
			inverse -> entry[row][column] *= factorReciprocal;
		}

		for(int anotherRow = 0; anotherRow < copiedMatrix.rowCount; anotherRow ++){
			if(anotherRow != row && copiedMatrix.entry[anotherRow][row] != 0){
				double anotherFactor = copiedMatrix.entry[anotherRow][row];

				for(int column = row; column < copiedMatrix.columnCount; column ++){
					copiedMatrix.entry[anotherRow][column] -= copiedMatrix.entry[row][column] * anotherFactor;
				}
				for(int column = 0; column < inverse -> columnCount; column ++){
					inverse -> entry[anotherRow][column] -= inverse -> entry[row][column] * anotherFactor;
				}	
			}
		}
	}

	matrixReleaseSpace(&copiedMatrix);
	return inverseExistent;
}

// Algorithm: Cholesky decomposition to avoid directly calculating matrix determinant where the overflow is likely to occur for large matrices
// \ln \det A = 2 \sum \ln diag(L) where A = L L^T, L is the cholesky factor
// The matrix as input should be positive-definite
double matrixCalculatePositiveDefiniteLogDeterminant(Matrix *matrix){
	if(matrix -> rowCount != matrix -> columnCount){
		return 0.0;
	}

	Matrix choleskyFactor;
	matrixInitialize(&choleskyFactor, matrix -> rowCount, matrix -> columnCount);
	double logDeterminant = 0.0;
	
	// Cholesky decomposition
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column <= row; column ++){
			double anotherFactorEntryDotProduct = vectorCalculateDotProduct(choleskyFactor.entry[row], choleskyFactor.entry[column], column);

			if(row == column){
				choleskyFactor.entry[row][column] = sqrt(matrix -> entry[row][column] - anotherFactorEntryDotProduct);
				logDeterminant += log(choleskyFactor.entry[row][column]);
			}
			else{
				choleskyFactor.entry[row][column] = (matrix -> entry[row][column] - anotherFactorEntryDotProduct) / choleskyFactor.entry[column][column];	
			}
		}
	}
	logDeterminant *= 2;

	matrixReleaseSpace(&choleskyFactor);
	return logDeterminant;
}

//==============================================================================================

void arrayInitialize(Array *array){
	array -> length = 0;
	array -> entry = (ArrayEntry*)malloc(sizeof(ArrayEntry));
}

void arrayReleaseSpace(Array *array){
	free(array -> entry);
	array -> entry = NULL;
}

void arrayAddEntry(Array *array, int key1, int key2, double value){
	array -> entry = (ArrayEntry*)realloc(array -> entry, sizeof(ArrayEntry) * (array -> length + 1));
	array -> entry[array -> length].key1 = key1;
	array -> entry[array -> length].key2 = key2;
	array -> entry[array -> length].value = value;
	array -> length += 1;
}

void arrayCopyFromList(Array *array, List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entry[row][j].key;
			double value = list -> entry[row][j].value;

			arrayAddEntry(array, row, column, value);
		}
	}
}

void arrayShuffle(Array *array){
	for(int s = 0; s < array -> length; s ++){
		int i = randomSampleInteger(0, array -> length - 1);
		int j = randomSampleInteger(0, array -> length - 1);

		ArrayEntry temp = array -> entry[i];
		array -> entry[i] = array -> entry[j];
		array -> entry[j] = temp;
	}
}

void arrayScan(Array *array, FILE *inputStream){
	int key1, key2;
	double value;
	char line[BUFFER_SIZE];

	while(fgets(line, BUFFER_SIZE, inputStream) != NULL){
		int readValueCount = sscanf(line, "%d%d%lf", &key1, &key2, &value);

		switch(readValueCount){
			case 3:
				arrayAddEntry(array, key1, key2, value);
				break;
			case 2:
				arrayAddEntry(array, key1, key2, 1);
				break;
		}
	}
}

void arrayPrint(Array *array, FILE *outputStream){
	for(int a = 0; a < array -> length; a ++){
		int key1 = array -> entry[a].key1;
		int key2 = array -> entry[a].key2;
		double value = array -> entry[a].value;

		fprintf(outputStream, "%d\t%d\t%.10e\n", key1, key2, value);
	}
}

void arraySort(Array *array, int (*compare)(const void *p, const void *q)){
	qsort(array -> entry, array -> length, sizeof(ArrayEntry), compare);
}

//==============================================================================================

void ratingReadHeader(char *ratingFilePath, int *rowCount, int *columnCount){
	FILE *inFile = fopen(ratingFilePath, "r");

	int status = fscanf(inFile, "%d%d", rowCount, columnCount);

	fclose(inFile);
}

void ratingReadArray(char *ratingFilePath, Array *ratingArray){
	FILE *inFile = fopen(ratingFilePath, "r");
	
	// Skip the header
	char line[BUFFER_SIZE];
	char *status = fgets(line, BUFFER_SIZE, inFile);
	arrayScan(ratingArray, inFile);

	fclose(inFile);
}

void ratingReadList(char *ratingFilePath, List *ratingList){
	FILE *inFile = fopen(ratingFilePath, "r");
	
	// Skip the header
	char line[BUFFER_SIZE];
	char *status = fgets(line, BUFFER_SIZE, inFile);
	listScan(ratingList, inFile);

	fclose(inFile);
}

void ratingReadMatrix(char *ratingFilePath, Matrix *ratingMatrix){
	FILE *inFile = fopen(ratingFilePath, "r");
	
	// Skip the header
	char line[BUFFER_SIZE];
	char *status = fgets(line, BUFFER_SIZE, inFile);
	matrixScan(ratingMatrix, inFile);

	fclose(inFile);
}

void ratingWriteMatrix(char *ratingFilePath, Matrix *ratingMatrix){
	FILE *outFile = fopen(ratingFilePath, "w");

	fprintf(outFile, "%d\t%d\n", ratingMatrix -> rowCount, ratingMatrix -> columnCount);
	matrixPrint(ratingMatrix, outFile);

	fclose(outFile);
}

void ratingWriteArray(char *ratingFilePath, Array *ratingArray, int key1Count, int key2Count){
	FILE *outFile = fopen(ratingFilePath, "w");

	fprintf(outFile, "%d\t%d\n", key1Count, key2Count);
	arrayPrint(ratingArray, outFile);

	fclose(outFile);
}

void ratingWriteList(char *ratingFilePath, List *ratingList, int columnCount){
	FILE *outFile = fopen(ratingFilePath, "w");

	fprintf(outFile, "%d\t%d\n", ratingList -> rowCount, columnCount);
	listPrint(ratingList, outFile);

	fclose(outFile);
}

void ratingNormalizeArrayByMean(Array *ratingArray){
	if(ratingArray -> length == 0){
		return;
	}

	double mean = 0.0;

	for(int a = 0; a < ratingArray -> length; a ++){
		double rating = ratingArray -> entry[a].value;

		mean += rating;
	}

	mean /= ratingArray -> length;

	for(int a = 0; a < ratingArray -> length; a ++){
		ratingArray -> entry[a].value -= mean;
	}
}

void ratingNormalizeListByMean(List *ratingList){
	double mean = 0;
	int count = 0;

	for(int user = 0; user < ratingList -> rowCount; user ++){
		for(int j = 0; j < ratingList -> columnCounts[user]; j ++){
			double rating = ratingList -> entry[user][j].value;
			mean += rating;
			count += 1;
		}
	}

	if(count == 0){
		return;
	}
	mean /= count;

	for(int user = 0; user < ratingList -> rowCount; user ++){
		for(int j = 0; j < ratingList -> columnCounts[user]; j ++){
			ratingList -> entry[user][j].value -= mean;
		}
	}
}

void ratingSplitArrayValidation(Array *ratingArray, Array *trainingRatingArray, Array* validationRatingArray, float validationRatio){
	for(int a = 0; a < ratingArray -> length; a ++){
		int user = ratingArray -> entry[a].key1;
		int item = ratingArray -> entry[a].key2;
		double rating = ratingArray -> entry[a].value;

		double probability = randomSampleStardardUniformVariable();
		if(probability < validationRatio){
			arrayAddEntry(validationRatingArray, user, item, rating);
		}
		else{
			arrayAddEntry(trainingRatingArray, user, item, rating);
		}
	}
}

void ratingSplitListValidation(List *ratingList, List *trainingRatingList, List* validationRatingList, float validationRatio){
	for(int user = 0; user < ratingList -> rowCount; user ++){
		for(int j = 0; j < ratingList -> columnCounts[user]; j ++){
			int item = ratingList -> entry[user][j].key;
			double rating = ratingList -> entry[user][j].value;

			double probability = randomSampleStardardUniformVariable();
			if(probability < validationRatio){
				listAddEntry(validationRatingList, user, item, rating);
			}
			else{
				listAddEntry(trainingRatingList, user, item, rating);
			}
		}
	}
}

//==============================================================================================

void optimizerInitialize(Optimizer *optimizer){
	optimizer -> epochCount = 1000;
	optimizer -> convergenceThreshold = 1e-4;
	optimizer -> decay = 0.95;
	optimizer -> conditionConstant = 1e-6;
}

double optimizerAdadeltaGetStep(Optimizer *optimizer, double gradient, double *squaredStep, double *secondMoment){
	*secondMoment = optimizer -> decay * (*secondMoment) + (1.0 - optimizer -> decay) * gradient * gradient;

	double step = sqrt((*squaredStep + optimizer -> conditionConstant) / (*secondMoment + optimizer -> conditionConstant)) * gradient;

	*squaredStep = optimizer -> decay * (*squaredStep) + (1.0 - optimizer -> decay) * step * step;

	return step;
}
