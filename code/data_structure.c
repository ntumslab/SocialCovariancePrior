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

double vectorCalculateDotProduct(double *vector1, double *vector2, int length){
	double dotProduct = 0.0;
	for(int d = 0; d < length; d ++){
		dotProduct += vector1[d] * vector2[d];
	}
	return dotProduct;
}
//==============================================================================================
void listInitialize(List *list, int rowCount){
	list -> rowCount = rowCount;
	list -> columnCounts = (int*)malloc(sizeof(int) * rowCount);
	list -> entries = (Dict**)malloc(sizeof(Dict*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		list -> columnCounts[row] = 0;
		list -> entries[row] = (Dict*)malloc(sizeof(Dict));
	}
}

void listReleaseSpace(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		free(list -> entries[row]);
	}
	free(list -> entries);
	free(list -> columnCounts);
	list -> entries = NULL;
	list -> columnCounts = NULL;
}

void listAddEntry(List *list, int row, int column, double value){
	int columnCount = list -> columnCounts[row];
	list -> entries[row] = (Dict*)realloc(list -> entries[row], sizeof(Dict) * (columnCount + 1));
	list -> entries[row][columnCount].key = column;
	list -> entries[row][columnCount].value = value;
	list -> columnCounts[row] += 1;
}

void listCopyEntries(List *source, List *target){
	for(int row = 0; row < source -> rowCount; row ++){
		for(int j = 0; j < source -> columnCounts[row]; j ++){
			int column = source -> entries[row][j].key;
			double value = source -> entries[row][j].value;
			listAddEntry(target, row, column, value);
		}
	}
}

void listSortRows(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		qsort(list -> entries[row], list -> columnCounts[row], sizeof(Dict), dictCompareAscendingKeys);
	}
}

int listCountEntries(List *list){
	int entryCount = 0;
	for(int row = 0; row < list -> rowCount; row ++){
		entryCount += list -> columnCounts[row];
	}
	return entryCount;
}

// returns a row vector sorted by keys (column index)
void listGetRowVector(List *list, int row, Dict **vectorPointer, int *vectorLength){
	*vectorLength = list -> columnCounts[row];
	*vectorPointer = (Dict*)realloc(*vectorPointer, sizeof(Dict) * list -> columnCounts[row]);

	for(int j = 0; j < list -> columnCounts[row]; j ++){
		(*vectorPointer)[j] = list -> entries[row][j];
	}

	qsort(*vectorPointer, *vectorLength, sizeof(Dict), dictCompareAscendingKeys);
}

// returns a column vectors sorted by keys (row index)
void listGetColumnVector(List *list, int column, Dict **vectorPointer, int *vectorLength){
	*vectorLength = 0;
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			if(list -> entries[row][j].key == column){
				*vectorPointer = (Dict*)realloc(*vectorPointer, sizeof(Dict) * (*vectorLength + 1));
				(*vectorPointer)[*vectorLength].key = row;
				(*vectorPointer)[*vectorLength].value = list -> entries[row][j].value;
				*vectorLength += 1;
				break;
			}
		}
	}
}

// rowVectors should contain at least the number of rows that list contains
void listGetAllRowVectors(List *list, List *rowVectors){
	for(int row = 0; row < list -> rowCount; row ++){
		rowVectors -> entries[row] = (Dict*)realloc(rowVectors -> entries[row], sizeof(Dict) * (list -> columnCounts[row]));
		rowVectors -> columnCounts[row] = list -> columnCounts[row];

		for(int j = 0; j < list -> columnCounts[row]; j ++){
			rowVectors -> entries[row][j].key = list -> entries[row][j].key;
			rowVectors -> entries[row][j].value = list -> entries[row][j].value;
		}
		qsort(rowVectors -> entries[row], rowVectors -> columnCounts[row], sizeof(Dict), dictCompareAscendingKeys);
	}
}

// columnVectors should contain at least the number of columns that list contains
void listGetAllColumnVectors(List *list, List *columnVectors){
	for(int row = 0; row < columnVectors -> rowCount; row ++){
		columnVectors -> columnCounts[row] = 0;
	}
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entries[row][j].key;
			double value = list -> entries[row][j].value;

			int count = columnVectors -> columnCounts[column];
			columnVectors -> entries[column] = (Dict*)realloc(columnVectors -> entries[column], sizeof(Dict) * (count + 1));
			columnVectors -> entries[column][count].key = row;
			columnVectors -> entries[column][count].value = value;
			columnVectors -> columnCounts[column] += 1;
		}
	}
	for(int row = 0; row < columnVectors -> rowCount; row ++){
		qsort(columnVectors -> entries[row], columnVectors -> columnCounts[row], sizeof(Dict), dictCompareAscendingKeys);
	}
}

// The size of array entryCount should be at least the number of rows of the list
void listCountRowEntries(List *list, int *entryCounts){
	for(int row = 0; row < list -> rowCount; row ++){
		entryCounts[row] = list -> columnCounts[row];
	}
}

// The size of array entryCount should be at least the number of columns of the list
void listCountColumnEntries(List *list, int *entryCounts, int columnCount){
	for(int column = 0; column < columnCount; column ++){
		entryCounts[column] = 0;
	}

	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entries[row][j].key;

			entryCounts[column] += 1;
		}
	}
}

void listPrint(List *list, FILE *outputStream){
	for(int row = 0; row < list -> rowCount; row ++){
		for(int j = 0; j < list -> columnCounts[row]; j ++){
			int column = list -> entries[row][j].key;
			double value = list -> entries[row][j].value;
			fprintf(outputStream, "%d\t%d\t%.10f\n", row, column, value);
		}
	}
}

void listScan(FILE *inputStream, List *list){
	int row, column;
	double value;
	char line[10000];
	while(fgets(line, 10000, inputStream) != NULL){
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
//==============================================================================================
void matrixInitialize(Matrix *matrix, int rowCount, int columnCount){
	matrix -> rowCount = rowCount;
	matrix -> columnCount = columnCount;
	matrix -> entries = (double**)malloc(sizeof(double*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		matrix -> entries[row] = (double*)malloc(sizeof(double) * columnCount);
	}	
}

void matrixReleaseSpace(Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		free(matrix -> entries[row]);
	}
	free(matrix -> entries);
	matrix -> entries = NULL;
}

void matrixAssignRandomValues(Matrix *matrix, double minValue, double maxValue){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			double uniform = (double)rand() / RAND_MAX;
			matrix -> entries[row][column] = minValue + (maxValue - minValue) * uniform;
		}
	}
}

void matrixCopyEntries(Matrix *source, Matrix *target){
	for(int row = 0; row < source -> rowCount; row ++){
		for(int column = 0; column < source -> columnCount; column ++){
			target -> entries[row][column] = source -> entries[row][column];
		}
	}
}

void matrixPrint(Matrix *matrix, FILE *outputStream){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			if(column > 0){
				fprintf(outputStream, "\t");
			}
			fprintf(outputStream, "%.10f", matrix -> entries[row][column]);
		}
		fprintf(outputStream, "\n");
	}
}

void matrixScan(FILE *inputStream, Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			int argumentCount = fscanf(inputStream, "%lf", &matrix -> entries[row][column]);
		}
	}	
}

void matrixSetIdentity(Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entries[row][column] = 0;
		}
		matrix -> entries[row][row] = 1;
	}
}

void matrixSetValue(Matrix *matrix, double value){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entries[row][column] = value;
		}
	}
}

void matrixMultiplyScalar(Matrix *matrix, double scalar){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entries[row][column] *= scalar;
		}
	}	
}

void matrixAddScalar(Matrix *matrix, double scalar){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			matrix -> entries[row][column] += scalar;
		}
	}	
}

double matrixCalculateSquareSum(Matrix *matrix){
	double squareSum = 0.0;

	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			double value = matrix -> entries[row][column];
			squareSum += value * value;
		}
	}

	return squareSum;
}

// The transpose should be of the same size (#rows and #columns) as the input matrix
void matrixGetTranspose(Matrix *matrix, Matrix *transpose){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			transpose -> entries[column][row] = matrix -> entries[row][column];
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
		if(copiedMatrix.entries[row][row] == 0){
			nonZeroRowFound = false;

			for(int anotherRow = row + 1; anotherRow < copiedMatrix.rowCount; anotherRow ++){
				if(copiedMatrix.entries[anotherRow][row] != 0){
					nonZeroRowFound = true;
					
					for(int column = row; column < copiedMatrix.columnCount; column ++){
						double temp1 = copiedMatrix.entries[row][column];
						copiedMatrix.entries[row][column] = copiedMatrix.entries[anotherRow][column];
						copiedMatrix.entries[anotherRow][column] = temp1;
					}
					for(int column = 0; column < inverse -> columnCount; column ++){
						double temp2 = inverse -> entries[row][column];
						inverse -> entries[row][column] = inverse -> entries[anotherRow][column];
						inverse -> entries[anotherRow][column] = temp2;
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
		double factorReciprocal = 1.0 / copiedMatrix.entries[row][row];
		for(int column = row; column < copiedMatrix.columnCount; column ++){
			copiedMatrix.entries[row][column] *= factorReciprocal;
		}
		for(int column = 0; column < copiedMatrix.columnCount; column ++){
			inverse -> entries[row][column] *= factorReciprocal;
		}

		for(int anotherRow = 0; anotherRow < copiedMatrix.rowCount; anotherRow ++){
			if(anotherRow != row && copiedMatrix.entries[anotherRow][row] != 0){
				double anotherFactor = copiedMatrix.entries[anotherRow][row];

				for(int column = row; column < copiedMatrix.columnCount; column ++){
					copiedMatrix.entries[anotherRow][column] -= copiedMatrix.entries[row][column] * anotherFactor;
				}
				for(int column = 0; column < inverse -> columnCount; column ++){
					inverse -> entries[anotherRow][column] -= inverse -> entries[row][column] * anotherFactor;
				}	
			}
		}
	}

	matrixReleaseSpace(&copiedMatrix);
	return inverseExistent;
}

//==============================================================================================

void ratingFetchUserItemCount(char *ratingFilePath, int *userCount, int *itemCount){
	FILE *inFile = fopen(ratingFilePath, "r");
	char line[10000];
	int lastUser = 0;
	int lastItem = 0;

	while(fgets(line, 10000, inFile)){
		int user, item;
		sscanf(line, "%d%d", &user, &item);
		if(lastUser < user){
			lastUser = user;
		}
		if(lastItem < item){
			lastItem = item;
		}
	}

	fclose(inFile);
	*userCount = lastUser + 1;
	*itemCount = lastItem + 1;
}

// User index [0, userCount - 1], item index [0, itemCount - 1];
void ratingReadFromFile(char *ratingFilePath, List *ratings){
	FILE *inFile = fopen(ratingFilePath, "r");
	listScan(inFile, ratings);
	fclose(inFile);
}

void ratingNormalizeByMean(List *ratings){
	double mean = 0;
	int count = 0;
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			double rating = ratings -> entries[user][j].value;
			mean += rating;
			count += 1;
		}
	}

	if(count == 0){
		return;
	}
	mean /= count;

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			ratings -> entries[user][j].value -= mean;
		}
	}
}
