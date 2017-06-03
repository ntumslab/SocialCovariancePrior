#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <tgmath.h>
#include "data_structure.h"

typedef struct MATRIX_FACTORIZATION{
	int latentFactorCount;							// K
	int maxIterationCount;
	double convergenceThreshold;
	int similaritySamplingCount;					// | F |
	int userCount;									// N
	int itemCount;									// M
	Matrix *userFactorVariationalMeanMatrix;		// \lambda_{U} \in R^{N \times K}
	Matrix *itemFactorVariationalMeanMatrix;		// \lambda_{V} \in R^{M \times K} 
	Matrix *userFactorVariationalVarianceMatrix;	// \gamma_{U} \in R^{N \times K}
	Matrix *itemFactorVariationalVarianceMatrix;	// \gamma_{V} \in R^{M \times K}
	double userBalanceParameter;					// b_{U}
	double itemBalanceParameter;					// b_{V}
	bool userExplicitSocialNetworkImported;
	bool itemExplicitSocialNetworkImported;
} MatrixFactorization;

double matrixFactorizationEvaluateRMSE(MatrixFactorization *model, List *ratings);

// Notation: 1 / \sigma_{U}^{2}
double matrixFactorizationEstimateUserFactorPrecision(MatrixFactorization *model){
	double userFactorCovariance = 0;

	for(int user = 0; user < model -> userCount; user ++){
		for(int k = 0; k < model -> latentFactorCount; k ++){
			double mean = model -> userFactorVariationalMeanMatrix -> entries[user][k];
			double variance = model -> userFactorVariationalVarianceMatrix -> entries[user][k];

			userFactorCovariance += mean * mean + variance;
		}
	}
	userFactorCovariance /= model -> userCount * model -> latentFactorCount;

	double userFactorPrecision = 1.0 / userFactorCovariance;
	return userFactorPrecision;
}

// Notation: 1 / \sigma_{V}^{2}
double matrixFactorizationEstimateItemFactorPrecision(MatrixFactorization *model){
	double itemFactorCovariance = 0;

	for(int item = 0; item < model -> itemCount; item ++){
		for(int k = 0; k < model -> latentFactorCount; k ++){
			double mean = model -> itemFactorVariationalMeanMatrix -> entries[item][k];
			double variance = model -> itemFactorVariationalVarianceMatrix -> entries[item][k];

			itemFactorCovariance += mean * mean + variance;
		}
	}
	itemFactorCovariance /= model -> itemCount * model -> latentFactorCount;

	double itemFactorPrecision = 1.0 / itemFactorCovariance;
	return itemFactorPrecision;
}

// Notation: 1 / \Lambda_{U}, \Lambda{Uif} in the implicit social network
void matrixFactorizationEstimateUserSimilarityMatrix(MatrixFactorization *model, Matrix *userSimilaritySumMatrix, Matrix *userSimilarityWeightedSumMatrix, Matrix *userFactorSimilarityRateVector){
	matrixSetValue(userSimilaritySumMatrix, 0);
	matrixSetValue(userSimilarityWeightedSumMatrix, 0);

	int userSamplingCount = (model -> similaritySamplingCount < model -> userCount - 1)? model -> similaritySamplingCount: model -> userCount - 1;
	double userSamplingRate = (double)userSamplingCount / (model -> userCount - 1);
	Matrix userFactorSimilarityMeanVector;
	matrixInitialize(&userFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&userFactorSimilarityMeanVector, 0);
	for(int user = 0; user < model -> userCount; user ++){
		for(int friend = 0; friend < model -> userCount; friend ++){
			if(user == friend || randomSampleStardardUniformVariable() > userSamplingRate){
				continue;
			}

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userMean = model -> userFactorVariationalMeanMatrix -> entries[user][k];
				double friendMean = model -> userFactorVariationalMeanMatrix -> entries[friend][k];
				double userVariance = model -> userFactorVariationalVarianceMatrix -> entries[user][k];
				double friendVariance = model -> userFactorVariationalVarianceMatrix -> entries[friend][k];

				double similarityRate = ((userMean - friendMean) * (userMean - friendMean) + userVariance + friendVariance + userFactorSimilarityRateVector -> entries[0][k]);
				double similarity = (model -> latentFactorCount + 1) / similarityRate;
				double normalizedSimilarity = similarity / userSamplingCount;
			
				userSimilaritySumMatrix -> entries[user][k] += normalizedSimilarity;
				userSimilaritySumMatrix -> entries[friend][k] += normalizedSimilarity;
				
				userSimilarityWeightedSumMatrix -> entries[user][k] += 
					normalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entries[friend][k];
				userSimilarityWeightedSumMatrix -> entries[friend][k] += 
					normalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entries[user][k];

				userFactorSimilarityMeanVector.entries[0][k] += normalizedSimilarity;
			}	
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		userFactorSimilarityMeanVector.entries[0][k] /= model -> userCount;
		userFactorSimilarityRateVector -> entries[0][k] = model -> latentFactorCount / userFactorSimilarityMeanVector.entries[0][k];
	}
	matrixReleaseSpace(&userFactorSimilarityMeanVector);
}

// Notation: 1 / \Lambda_{V}, \Lambda{Vjg} in the implicit social network
void matrixFactorizationEstimateItemSimilarityMatrix(MatrixFactorization *model, Matrix *itemSimilaritySumMatrix, Matrix *itemSimilarityWeightedSumMatrix, Matrix *itemFactorSimilarityRateVector){
	matrixSetValue(itemSimilaritySumMatrix, 0);
	matrixSetValue(itemSimilarityWeightedSumMatrix, 0);

	int itemSamplingCount = (model -> similaritySamplingCount < model -> itemCount - 1)? model -> similaritySamplingCount: model -> itemCount - 1;
	double itemSamplingRate = (double)itemSamplingCount / (model -> itemCount - 1);
	Matrix itemFactorSimilarityMeanVector;
	matrixInitialize(&itemFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&itemFactorSimilarityMeanVector, 0);
	for(int item = 0; item < model -> itemCount; item ++){
		for(int friend = 0; friend < model -> itemCount; friend ++){
			if(item == friend || randomSampleStardardUniformVariable() > itemSamplingRate){
				continue;
			}

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double itemMean = model -> itemFactorVariationalMeanMatrix -> entries[item][k];
				double friendMean = model -> itemFactorVariationalMeanMatrix -> entries[friend][k];
				double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entries[item][k];
				double friendVariance = model -> itemFactorVariationalVarianceMatrix -> entries[friend][k];

				double similarityRate = ((itemMean - friendMean) * (itemMean - friendMean) + itemVariance + friendVariance + itemFactorSimilarityRateVector -> entries[0][k]);
				double similarity = (model -> latentFactorCount + 1) / similarityRate;
				double normalizedSimilarity = similarity / itemSamplingCount;
			
				itemSimilaritySumMatrix -> entries[item][k] += normalizedSimilarity;
				itemSimilaritySumMatrix -> entries[friend][k] += normalizedSimilarity;
				
				itemSimilarityWeightedSumMatrix -> entries[item][k] += 
					normalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entries[friend][k];
				itemSimilarityWeightedSumMatrix -> entries[friend][k] += 
					normalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entries[item][k];

				itemFactorSimilarityMeanVector.entries[0][k] += normalizedSimilarity;
			}
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		itemFactorSimilarityMeanVector.entries[0][k] /= model -> itemCount;
		itemFactorSimilarityRateVector -> entries[0][k] = model -> latentFactorCount / itemFactorSimilarityMeanVector.entries[0][k];
	}
	matrixReleaseSpace(&itemFactorSimilarityMeanVector);
}

// Notation: 1 / \Lambda_{U}, \Lambda{Uif} in the explicit social network
void matrixFactorizationEstimateUserSparseSimilarityMatrix(MatrixFactorization *model, Matrix *userSimilaritySumMatrix, Matrix *userSimilarityWeightedSumMatrix, Matrix *userFactorSimilarityRateVector, List *userSocialNetwork){
	matrixSetValue(userSimilaritySumMatrix, 0);
	matrixSetValue(userSimilarityWeightedSumMatrix, 0);

	Matrix userFactorSimilarityMeanVector;
	matrixInitialize(&userFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&userFactorSimilarityMeanVector, 0);
	for(int user = 0; user < model -> userCount; user ++){
		for(int f = 0; f < userSocialNetwork -> columnCounts[user]; f ++){
			int friend = userSocialNetwork -> entries[user][f].key;

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userMean = model -> userFactorVariationalMeanMatrix -> entries[user][k];
				double friendMean = model -> userFactorVariationalMeanMatrix -> entries[friend][k];
				double userVariance = model -> userFactorVariationalVarianceMatrix -> entries[user][k];
				double friendVariance = model -> userFactorVariationalVarianceMatrix -> entries[friend][k];

				double similarityRate = ((userMean - friendMean) * (userMean - friendMean) + userVariance + friendVariance + userFactorSimilarityRateVector -> entries[0][k]);
				double similarity = (model -> latentFactorCount + 1) / similarityRate;
				
				double userNormalizedSimilarity = similarity / userSocialNetwork -> columnCounts[user];
				userSimilaritySumMatrix -> entries[user][k] += userNormalizedSimilarity;
				userSimilarityWeightedSumMatrix -> entries[user][k] += 
					userNormalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entries[friend][k];

				if(userSocialNetwork -> columnCounts[friend] > 0){
					double friendNormalizedSimilarity = similarity / userSocialNetwork -> columnCounts[friend];
					userSimilaritySumMatrix -> entries[friend][k] += friendNormalizedSimilarity;
					userSimilarityWeightedSumMatrix -> entries[friend][k] += 
						friendNormalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entries[user][k];
				}

				userFactorSimilarityMeanVector.entries[0][k] += userNormalizedSimilarity;
			}
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		userFactorSimilarityMeanVector.entries[0][k] /= model -> userCount;
		userFactorSimilarityRateVector -> entries[0][k] = model -> latentFactorCount / userFactorSimilarityMeanVector.entries[0][k];
	}
	matrixReleaseSpace(&userFactorSimilarityMeanVector);
}

// Notation: 1 / \Lambda_{V}, \Lambda{Vjg} in the explicit social network
void matrixFactorizationEstimateItemSparseSimilarityMatrix(MatrixFactorization *model, Matrix *itemSimilaritySumMatrix, Matrix *itemSimilarityWeightedSumMatrix, Matrix *itemFactorSimilarityRateVector, List *itemSocialNetwork){
	matrixSetValue(itemSimilaritySumMatrix, 0);
	matrixSetValue(itemSimilarityWeightedSumMatrix, 0);

	Matrix itemFactorSimilarityMeanVector;
	matrixInitialize(&itemFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&itemFactorSimilarityMeanVector, 0);
	for(int item = 0; item < model -> itemCount; item ++){
		for(int f = 0; f < itemSocialNetwork -> columnCounts[item]; f ++){
			int friend = itemSocialNetwork -> entries[item][f].key;
			
			for(int k = 0; k < model -> latentFactorCount; k ++){
				double itemMean = model -> itemFactorVariationalMeanMatrix -> entries[item][k];
				double friendMean = model -> itemFactorVariationalMeanMatrix -> entries[friend][k];
				double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entries[item][k];
				double friendVariance = model -> itemFactorVariationalVarianceMatrix -> entries[friend][k];

				double similarityRate = ((itemMean - friendMean) * (itemMean - friendMean) + itemVariance + friendVariance + itemFactorSimilarityRateVector -> entries[0][k]);
				double similarity = (model -> latentFactorCount + 1) / similarityRate;
				
				double itemNormalizedSimilarity = similarity / itemSocialNetwork -> columnCounts[item];
				itemSimilaritySumMatrix -> entries[item][k] += itemNormalizedSimilarity;
				itemSimilarityWeightedSumMatrix -> entries[item][k] += 
					itemNormalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entries[friend][k];

				if(itemSocialNetwork -> columnCounts[friend] > 0){
					double friendNormalizedSimilarity = similarity / itemSocialNetwork -> columnCounts[friend];
					itemSimilaritySumMatrix -> entries[friend][k] += friendNormalizedSimilarity;
					itemSimilarityWeightedSumMatrix -> entries[friend][k] += 
						friendNormalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entries[item][k];
				}

				itemFactorSimilarityMeanVector.entries[0][k] += itemNormalizedSimilarity;
			}	
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		itemFactorSimilarityMeanVector.entries[0][k] /= model -> itemCount;
		itemFactorSimilarityRateVector -> entries[0][k] = model -> latentFactorCount / itemFactorSimilarityMeanVector.entries[0][k];
	}
	matrixReleaseSpace(&itemFactorSimilarityMeanVector);
}

// Notation: 1 / \sigma_{R}^{2}
double matrixFactorizationEstimateRatingPrecision(MatrixFactorization *model, List *ratings){
	double ratingVariance = 0.0;
	int ratingCount = listCountEntries(ratings);

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double rating = ratings -> entries[user][j].value;

			double variationalMeanDotProduct = 0.0;
			double variationalVarianceDotProduct = 0.0;
			double variationalUserMeanSquareItemVarianceDotProduct = 0.0;
			double variationalItemMeanSquareUserVarianceDotProduct = 0.0;
			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userMean = model -> userFactorVariationalMeanMatrix -> entries[user][k];
				double itemMean = model -> itemFactorVariationalMeanMatrix -> entries[item][k];
				double userVariance = model -> userFactorVariationalVarianceMatrix -> entries[user][k];
				double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entries[item][k];

				variationalMeanDotProduct += userMean * itemMean;

				variationalVarianceDotProduct += userVariance * itemVariance;

				variationalUserMeanSquareItemVarianceDotProduct += userMean * userMean * itemVariance;
				
				variationalItemMeanSquareUserVarianceDotProduct += itemMean * itemMean * userVariance;
			}

			ratingVariance += rating * rating
				- 2 * rating * variationalMeanDotProduct
				+ variationalMeanDotProduct * variationalMeanDotProduct
				+ variationalVarianceDotProduct
				+ variationalUserMeanSquareItemVarianceDotProduct
				+ variationalItemMeanSquareUserVarianceDotProduct;
		}
	}
	ratingVariance /= ratingCount;
	double ratingPrecision = 1.0 / ratingVariance;
	return ratingPrecision;
}

// Notation: \lambda_{Ui} and \gamma_{Ui}
void matrixFactorizationEstimateUserFactorVariationalMeanVarianceMatrices(MatrixFactorization *model, List *userRatings,
		double userFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *userSimilarityWeightedSumMatrix,
		double ratingPrecision){
	
	for(int user = 0; user < model -> userCount; user ++){
		Matrix momentMatrix;
		matrixInitialize(&momentMatrix, model -> latentFactorCount, model -> latentFactorCount);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			for(int l = 0; l <= k; l ++){
				double moment = 0.0;
				for(int j = 0; j < userRatings -> columnCounts[user]; j ++){
					int item = userRatings -> entries[user][j].key;

					moment += model -> itemFactorVariationalMeanMatrix -> entries[item][k] * model -> itemFactorVariationalMeanMatrix -> entries[item][l];
					if(k == l){
						moment += model -> itemFactorVariationalVarianceMatrix -> entries[item][k];
					}
				}
				moment *= ratingPrecision;
				
				if(k == l){
					moment += userFactorPrecision * (1.0 - model -> userBalanceParameter);
				}
				
				if(k == l){
					moment += userSimilaritySumMatrix -> entries[user][k] * model -> userBalanceParameter;
				}
				
				momentMatrix.entries[k][l] = moment;
				momentMatrix.entries[l][k] = moment;
			}
		}

		Matrix momentInverseMatrix;
		matrixInitialize(&momentInverseMatrix, model -> latentFactorCount, model -> latentFactorCount);
		matrixGetInverse(&momentMatrix, &momentInverseMatrix);
		matrixReleaseSpace(&momentMatrix);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			model -> userFactorVariationalVarianceMatrix -> entries[user][k] = momentInverseMatrix.entries[k][k];
		}
		// =================================
		
		Matrix weightedRatingVector;
		matrixInitialize(&weightedRatingVector, 1, model -> latentFactorCount);

		for(int k = 0; k < model -> latentFactorCount; k ++){
			double weightedRating = 0.0;
			for(int j = 0; j < userRatings -> columnCounts[user]; j ++){
				int item = userRatings -> entries[user][j].key;
				double rating = userRatings -> entries[user][j].value;

				weightedRating += rating * model -> itemFactorVariationalMeanMatrix -> entries[item][k];
			}
			weightedRating *= ratingPrecision;

			weightedRating += userSimilarityWeightedSumMatrix -> entries[user][k] * model -> userBalanceParameter;

			weightedRatingVector.entries[0][k] = weightedRating;
		}
		
		// =================================

		for(int k = 0; k  < model -> latentFactorCount; k ++){
			model -> userFactorVariationalMeanMatrix -> entries[user][k] = vectorCalculateDotProduct(
					momentInverseMatrix.entries[k], weightedRatingVector.entries[0], model -> latentFactorCount);
		}

		matrixReleaseSpace(&momentInverseMatrix);
		matrixReleaseSpace(&weightedRatingVector);
	}
}

// Notation: \lambda_{Vj} and \gamma_{Vj}
void matrixFactorizationEstimateItemFactorVariationalMeanVarianceMatrices(MatrixFactorization *model, List *itemRatings,
		double itemFactorPrecision,
		Matrix *itemSimilaritySumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		double ratingPrecision){

	for(int item = 0; item < model -> itemCount; item ++){
		Matrix momentMatrix;
		matrixInitialize(&momentMatrix, model -> latentFactorCount, model -> latentFactorCount);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			for(int l = 0; l <= k; l ++){
				double moment = 0.0;
				for(int j = 0; j < itemRatings -> columnCounts[item]; j ++){
					int user = itemRatings -> entries[item][j].key;

					moment += model -> userFactorVariationalMeanMatrix -> entries[user][k] * model -> userFactorVariationalMeanMatrix -> entries[user][l];
					if(k == l){
						moment += model -> userFactorVariationalVarianceMatrix -> entries[user][k];
					}
				}
				moment *= ratingPrecision;
				
				if(k == l){
					moment += itemFactorPrecision * (1.0 - model -> itemBalanceParameter);
				}
				
				if(k == l){
					moment += itemSimilaritySumMatrix -> entries[item][k] * model -> itemBalanceParameter;
				}
				
				momentMatrix.entries[k][l] = moment;
				momentMatrix.entries[l][k] = moment;
			}
		}

		Matrix momentInverseMatrix;
		matrixInitialize(&momentInverseMatrix, model -> latentFactorCount, model -> latentFactorCount);
		matrixGetInverse(&momentMatrix, &momentInverseMatrix);
		matrixReleaseSpace(&momentMatrix);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			model -> itemFactorVariationalVarianceMatrix -> entries[item][k] = momentInverseMatrix.entries[k][k];
		}
		// =================================
		
		Matrix weightedRatingVector;
		matrixInitialize(&weightedRatingVector, 1, model -> latentFactorCount);

		for(int k = 0; k < model -> latentFactorCount; k ++){
			double weightedRating = 0.0;
			for(int i = 0; i < itemRatings -> columnCounts[item]; i ++){
				int user = itemRatings -> entries[item][i].key;
				double rating = itemRatings -> entries[item][i].value;

				weightedRating += rating * model -> userFactorVariationalMeanMatrix -> entries[user][k];
			}
			weightedRating *= ratingPrecision;
			
			weightedRating += itemSimilarityWeightedSumMatrix -> entries[item][k] * model -> itemBalanceParameter;
			
			weightedRatingVector.entries[0][k] = weightedRating;
		}
		
		// =================================

		for(int k = 0; k  < model -> latentFactorCount; k ++){
			model -> itemFactorVariationalMeanMatrix -> entries[item][k] = vectorCalculateDotProduct(
					momentInverseMatrix.entries[k], weightedRatingVector.entries[0], model -> latentFactorCount);
		}

		matrixReleaseSpace(&momentInverseMatrix);
		matrixReleaseSpace(&weightedRatingVector);

	}
}

void matrixFactorizationRunVMStep(MatrixFactorization *model, List *ratings,
		double *userFactorPrecision, double *itemFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *itemSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		Matrix *userFactorSimilarityRateVector, Matrix *itemFactorSimilarityRateVector,
		double *ratingPrecision,
		List *userSocialNetwork, List *itemSocialNetwork){
	
	*ratingPrecision = matrixFactorizationEstimateRatingPrecision(model, ratings);
	
	if(model -> userBalanceParameter < 1){
		*userFactorPrecision = matrixFactorizationEstimateUserFactorPrecision(model);
	}

	if(model -> itemBalanceParameter < 1){
		*itemFactorPrecision = matrixFactorizationEstimateItemFactorPrecision(model);
	}
	
	if(model -> userBalanceParameter > 0){
		if(model -> userExplicitSocialNetworkImported){
			matrixFactorizationEstimateUserSparseSimilarityMatrix(model, userSimilaritySumMatrix, userSimilarityWeightedSumMatrix, userFactorSimilarityRateVector, userSocialNetwork);
		}else{
			matrixFactorizationEstimateUserSimilarityMatrix(model, userSimilaritySumMatrix, userSimilarityWeightedSumMatrix, userFactorSimilarityRateVector);
		}
	}

	if(model -> itemBalanceParameter > 0){
		if(model -> itemExplicitSocialNetworkImported){
			matrixFactorizationEstimateItemSparseSimilarityMatrix(model, itemSimilaritySumMatrix, itemSimilarityWeightedSumMatrix, itemFactorSimilarityRateVector, itemSocialNetwork);
		}else{
			matrixFactorizationEstimateItemSimilarityMatrix(model, itemSimilaritySumMatrix, itemSimilarityWeightedSumMatrix, itemFactorSimilarityRateVector);
		}
	}
}

void matrixFactorizationRunVEStep(MatrixFactorization *model, List *userRatings, List *itemRatings,
		double userFactorPrecision, double itemFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *itemSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		double ratingPrecision){

	matrixFactorizationEstimateUserFactorVariationalMeanVarianceMatrices(model, userRatings,
		userFactorPrecision, 
		userSimilaritySumMatrix, userSimilarityWeightedSumMatrix,
		ratingPrecision);

	matrixFactorizationEstimateItemFactorVariationalMeanVarianceMatrices(model, itemRatings, 
		itemFactorPrecision,
		itemSimilaritySumMatrix, itemSimilarityWeightedSumMatrix,
		ratingPrecision);
}

void matrixFactorizationLearn(MatrixFactorization *model, List *trainingRatings, List *validationRatings, List *userSocialNetwork, List *itemSocialNetwork){
	List *userRatings = trainingRatings;
	List itemRatings;
	listInitialize(&itemRatings, model -> itemCount);
	
	listSortRows(userRatings);
	listGetAllColumnVectors(trainingRatings, &itemRatings);

	double ratingPrecision = 1;
	
	double userFactorPrecision = 1;
	double itemFactorPrecision = 1;

	Matrix userFactorSimilarityRateVector;
	Matrix itemFactorSimilarityRateVector;
	matrixInitialize(&userFactorSimilarityRateVector, 1, model -> latentFactorCount);
	matrixInitialize(&itemFactorSimilarityRateVector, 1, model -> latentFactorCount);
	matrixSetValue(&userFactorSimilarityRateVector, 1);
	matrixSetValue(&itemFactorSimilarityRateVector, 1);
	
	Matrix userSimilaritySumMatrix;
	Matrix itemSimilaritySumMatrix;
	matrixInitialize(&userSimilaritySumMatrix, model -> userCount, model -> latentFactorCount);
	matrixInitialize(&itemSimilaritySumMatrix, model -> itemCount, model -> latentFactorCount);
	matrixAssignRandomValues(&userSimilaritySumMatrix, 0, 1);
	matrixAssignRandomValues(&itemSimilaritySumMatrix, 0, 1);
	
	Matrix userSimilarityWeightedSumMatrix;
	Matrix itemSimilarityWeightedSumMatrix;	
	matrixInitialize(&userSimilarityWeightedSumMatrix, model -> userCount, model -> latentFactorCount);
	matrixInitialize(&itemSimilarityWeightedSumMatrix, model -> itemCount, model -> latentFactorCount);
	matrixAssignRandomValues(&userSimilarityWeightedSumMatrix, 0, 1);
	matrixAssignRandomValues(&itemSimilarityWeightedSumMatrix, 0, 1);
	
	matrixAssignRandomValues(model -> userFactorVariationalMeanMatrix, 0, 1);
	matrixAssignRandomValues(model -> itemFactorVariationalMeanMatrix, 0, 1);
	matrixAssignRandomValues(model -> userFactorVariationalVarianceMatrix, 0, 1);
	matrixAssignRandomValues(model -> itemFactorVariationalVarianceMatrix, 0, 1);
	
	double lastValidationCost = DBL_MAX;
	for(int epoch = 0; epoch < model -> maxIterationCount; epoch ++){
		matrixFactorizationRunVMStep(model, trainingRatings,
			&userFactorPrecision, &itemFactorPrecision,
			&userSimilaritySumMatrix, &itemSimilaritySumMatrix,
			&userSimilarityWeightedSumMatrix, &itemSimilarityWeightedSumMatrix,
			&userFactorSimilarityRateVector, &itemFactorSimilarityRateVector,
			&ratingPrecision,
			userSocialNetwork, itemSocialNetwork);

		matrixFactorizationRunVEStep(model, userRatings, &itemRatings,
			userFactorPrecision, itemFactorPrecision,
			&userSimilaritySumMatrix, &itemSimilaritySumMatrix,
			&userSimilarityWeightedSumMatrix, &itemSimilarityWeightedSumMatrix,
			ratingPrecision);	

		double validationCost = matrixFactorizationEvaluateRMSE(model, validationRatings);
		printf("Iteration %4d\tValidation RMSE %f\tRMSE Descent %f\n", epoch + 1, validationCost, (epoch > 0) ? lastValidationCost - validationCost : 0.0);

		if(epoch > 0 && lastValidationCost - validationCost < model -> convergenceThreshold){
			break;
		}
		lastValidationCost = validationCost;
	}

	matrixReleaseSpace(&userSimilaritySumMatrix);
	matrixReleaseSpace(&itemSimilaritySumMatrix);
	matrixReleaseSpace(&userSimilarityWeightedSumMatrix);
	matrixReleaseSpace(&itemSimilarityWeightedSumMatrix);
	matrixReleaseSpace(&userFactorSimilarityRateVector);
	matrixReleaseSpace(&itemFactorSimilarityRateVector);
	
	listReleaseSpace(&itemRatings);
}

double matrixFactorizationPredict(MatrixFactorization *model, int user, int item){
	double predictedRating = vectorCalculateDotProduct(
			model -> userFactorVariationalMeanMatrix -> entries[user], model -> itemFactorVariationalMeanMatrix -> entries[item],
			model -> latentFactorCount);
	return predictedRating;
}

double matrixFactorizationEvaluateRMSE(MatrixFactorization *model, List *ratings){
	double rmse = 0.0;
	int totalRatingCount = listCountEntries(ratings);

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = matrixFactorizationPredict(model, user, item);

			double difference = predictedRating - trueRating;
			rmse += difference * difference;
		}
	}

	rmse = sqrt(rmse / totalRatingCount);
	return rmse;
}

double matrixFactorizationEvaluate(MatrixFactorization *model, List *ratings, int evaluationType){
	switch(evaluationType){
		case 1:
			return matrixFactorizationEvaluateRMSE(model, ratings);
	}
}
//==============================================================================================
typedef struct CROSS_VALIDATION{
	int foldCount;
	int evaluationTypeCount;
	int *evaluationTypes;
	int trainingFoldCount;
} CrossValidation;

void crossValidationGroupRatings(List *ratings, int foldCount, List *groupMarkers){
	for(int user = 0; user < ratings -> rowCount; user ++){
		groupMarkers -> entries[user] = (Dict*)realloc(groupMarkers -> entries[user], sizeof(Dict) * ratings -> columnCounts[user]);
		groupMarkers -> columnCounts[user] = ratings -> columnCounts[user];

		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			groupMarkers -> entries[user][j].key = item;
			groupMarkers -> entries[user][j].value = randomSampleInteger(0, foldCount - 1);
		}
	}
}

void crossValidationSplitRatings(List *ratings, List *groupMarkers, List *trainingRatings, List *validationRatings, int validationGroup){
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double rating = ratings -> entries[user][j].value;
			int group = groupMarkers -> entries[user][j].value;

			if(group == validationGroup){
				listAddEntry(validationRatings, user, item, rating);
			}
			else{
				listAddEntry(trainingRatings, user, item, rating);
			}
		}
	}
}

void crossValidationRun(CrossValidation *validation, MatrixFactorization *model, List *ratings, List *userSocialNetwork, List *itemSocialNetwork){
	int *userEntryCounts = (int*)malloc(sizeof(int) * model -> userCount);
	int *itemEntryCounts = (int*)malloc(sizeof(int) * model -> itemCount);

	List groupMarkers;
	listInitialize(&groupMarkers, ratings -> rowCount);
	crossValidationGroupRatings(ratings, validation -> foldCount, &groupMarkers);

	double *performanceMeans = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);
	double *performanceVariances = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);

	for(int e = 0; e < validation -> evaluationTypeCount; e ++){
		performanceMeans[e] = 0.0;
		performanceVariances[e] = 0.0;
	}

	for(int validedFold = 0; validedFold < validation -> foldCount; validedFold ++){
		List trainingRatings, validationRatings;
		listInitialize(&trainingRatings, ratings -> rowCount);
		listInitialize(&validationRatings, ratings -> rowCount);
		crossValidationSplitRatings(ratings, &groupMarkers, &trainingRatings, &validationRatings, validedFold);
		
		List trainingGroupMarkers, trainingTrainRatings, trainingValidRatings;
		listInitialize(&trainingGroupMarkers, ratings -> rowCount);
		listInitialize(&trainingTrainRatings, ratings -> rowCount);
		listInitialize(&trainingValidRatings, ratings -> rowCount);
		crossValidationGroupRatings(&trainingRatings, validation -> trainingFoldCount, &trainingGroupMarkers);
		crossValidationSplitRatings(&trainingRatings, &trainingGroupMarkers, &trainingTrainRatings, &trainingValidRatings, 0);

		matrixFactorizationLearn(model, &trainingTrainRatings, &trainingValidRatings, userSocialNetwork, itemSocialNetwork);
		
		printf("\t========\n");
		printf("\tCross validation %d\n", validedFold + 1);
		for(int e = 0; e < validation -> evaluationTypeCount; e ++){	
			double performance = matrixFactorizationEvaluate(model, &validationRatings, validation -> evaluationTypes[e]);
			performanceMeans[e] += performance;
			performanceVariances[e] += performance * performance;
			printf("\t\tRMSE %f\n", performance);
		}
		printf("\t========\n");

		listReleaseSpace(&trainingRatings);
		listReleaseSpace(&validationRatings);
		listReleaseSpace(&trainingGroupMarkers);
		listReleaseSpace(&trainingTrainRatings);
		listReleaseSpace(&trainingValidRatings);
	}

	double wholePerformanceValue = 0.0;
	for(int e = 0; e < validation -> evaluationTypeCount; e ++){	
		performanceMeans[e] /= validation -> foldCount;
		performanceVariances[e] = performanceVariances[e] / validation -> foldCount - performanceMeans[e] * performanceMeans[e];

		printf("\tMean of RMSE %f\n", performanceMeans[e]);
		printf("\tVariance of RMSE %e\n", performanceVariances[e]);

	}

	listReleaseSpace(&groupMarkers);
	free(performanceMeans);
	free(performanceVariances);
}

//==============================================================================================
int main(int argc, char *argv[]){
	srand(time(NULL));

	char *ratingFilePath = argv[1];

	int latentFactorCount = atoi(argv[2]);
	
	double userBalanceParameter;
	sscanf(argv[3], "%lf", &userBalanceParameter);
	double itemBalanceParameter;
	sscanf(argv[4], "%lf", &itemBalanceParameter);

	int userSocialNetworkImported = atoi(argv[5]);
	int itemSocialNetworkImported = atoi(argv[6]);
	
	char *userSocialNetworkFilePath = argv[7];
	char *itemSocialNetworkFilePath = argv[8];
	
	int userCount;
	int itemCount;
	ratingFetchUserItemCount(ratingFilePath, &userCount, &itemCount);
	
	// Read rating data
	List ratings;
	listInitialize(&ratings, userCount);
	ratingReadFromFile(ratingFilePath, &ratings);
	listSortRows(&ratings);
	ratingNormalizeByMean(&ratings);
	printf("%d users, %d items\n", userCount, itemCount);

	// Read user social network
	List userSocialNetwork;
	listInitialize(&userSocialNetwork, userCount);
	if(userSocialNetworkImported > 0){
		FILE *inFile = fopen(userSocialNetworkFilePath, "r");
		listScan(inFile, &userSocialNetwork);
		listSortRows(&userSocialNetwork);
		fclose(inFile);
	}

	// Read item social network	
	List itemSocialNetwork;
	listInitialize(&itemSocialNetwork, itemCount);
	if(itemSocialNetworkImported > 0){
		FILE *inFile = fopen(itemSocialNetworkFilePath, "r");
		listScan(inFile, &itemSocialNetwork);
		listSortRows(&itemSocialNetwork);
		fclose(inFile);
	}
	
	// Set Social Covariance Prior parameters
	MatrixFactorization mf = {
		.latentFactorCount = latentFactorCount,
		.maxIterationCount = 1000,
		.convergenceThreshold = 1e-4,
		.similaritySamplingCount = 100,
		.userCount = userCount,
		.itemCount = itemCount,
		.userBalanceParameter = userBalanceParameter,
		.itemBalanceParameter = itemBalanceParameter,
		.userExplicitSocialNetworkImported = userSocialNetworkImported > 0,
		.itemExplicitSocialNetworkImported = itemSocialNetworkImported > 0
	};
	Matrix userFactorVariationalMeanMatrix;
	Matrix itemFactorVariationalMeanMatrix;
	Matrix userFactorVariationalVarianceMatrix;
	Matrix itemFactorVariationalVarianceMatrix;

	matrixInitialize(&userFactorVariationalMeanMatrix, userCount, mf.latentFactorCount);
	matrixInitialize(&itemFactorVariationalMeanMatrix, itemCount, mf.latentFactorCount);
	matrixInitialize(&userFactorVariationalVarianceMatrix, userCount, mf.latentFactorCount);
	matrixInitialize(&itemFactorVariationalVarianceMatrix, itemCount, mf.latentFactorCount);

	mf.userFactorVariationalMeanMatrix = &userFactorVariationalMeanMatrix;
	mf.itemFactorVariationalMeanMatrix = &itemFactorVariationalMeanMatrix;
	mf.userFactorVariationalVarianceMatrix = &userFactorVariationalVarianceMatrix;
	mf.itemFactorVariationalVarianceMatrix = &itemFactorVariationalVarianceMatrix;

	printf("User variational mean matrix %d %d\n", userFactorVariationalMeanMatrix.rowCount, userFactorVariationalMeanMatrix.columnCount);
	printf("Item variational mean matrix %d %d\n", itemFactorVariationalMeanMatrix.rowCount, itemFactorVariationalMeanMatrix.columnCount);
	printf("User variational mean matrix %d %d\n", userFactorVariationalVarianceMatrix.rowCount, userFactorVariationalVarianceMatrix.columnCount);
	printf("Item variational mean matrix %d %d\n", itemFactorVariationalVarianceMatrix.rowCount, itemFactorVariationalVarianceMatrix.columnCount);
	
	// Run cross validation
	CrossValidation cv = {
		.foldCount = 5,
		.evaluationTypeCount = 1,
		.trainingFoldCount = 10
	};
	cv.evaluationTypes = (int*)malloc(sizeof(int) * cv.evaluationTypeCount);
	for(int e = 0; e < cv.evaluationTypeCount; e ++){
		cv.evaluationTypes[e] = e + 1;
	}
	crossValidationRun(&cv, &mf, &ratings, &userSocialNetwork, &itemSocialNetwork);
	free(cv.evaluationTypes);

	// Release space
	listReleaseSpace(&ratings);
	listReleaseSpace(&userSocialNetwork);
	listReleaseSpace(&itemSocialNetwork);
	matrixReleaseSpace(&userFactorVariationalMeanMatrix);
	matrixReleaseSpace(&itemFactorVariationalMeanMatrix);
	matrixReleaseSpace(&userFactorVariationalVarianceMatrix);
	matrixReleaseSpace(&itemFactorVariationalVarianceMatrix);

	printf("OK\n");
	
	return 0;
}
