#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include "data_structure.h"
// #define SCP_PARALLEL_OPEN

// Social Covariance Prior

typedef struct SOCIAL_COVARIANCE_PRIOR{
	int latentFactorCount;							// K
	int similaritySamplingCount;					// | F |
	int userCount;									// N
	int itemCount;									// M
	Matrix *userFactorVariationalMeanMatrix;		// \lambda_{U} \in R^{N \times K}
	Matrix *itemFactorVariationalMeanMatrix;		// \lambda_{V} \in R^{M \times K} 
	Matrix *userFactorVariationalVarianceMatrix;	// \gamma_{U} \in R^{N \times K}
	Matrix *itemFactorVariationalVarianceMatrix;	// \gamma_{V} \in R^{M \times K}
	double userFactorRegularizationRatio;			// 1 - b_{U}
	double itemFactorRegularizationRatio;			// 1 - b_{V}
	double userFriendRegularizationRatio;			// b_{U}
	double itemFriendRegularizationRatio;			// b_{V}
	bool userFriendImported;
	bool itemFriendImported;
	bool userFriendSimilarityClosed;
	bool itemFriendSimilarityClosed;
} SocialCovariancePrior;

double matrixFactorizationPredict(SocialCovariancePrior *model, int user, int item){
	return vectorCalculateDotProduct(
			model -> userFactorVariationalMeanMatrix -> entry[user], model -> itemFactorVariationalMeanMatrix -> entry[item],
			model -> latentFactorCount);
}

double matrixFactorizationEvaluateRMSE(SocialCovariancePrior *model, List *ratingList){
	double rmse = 0.0;
	int count = 0;

#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for reduction(+:rmse, count)
#endif
	for(int user = 0; user < ratingList -> rowCount; user ++){
		for(int j = 0; j < ratingList -> columnCounts[user]; j ++){	
			int item = ratingList -> entry[user][j].key;
			double rating = ratingList -> entry[user][j].value;

			double prediction = matrixFactorizationPredict(model, user, item);
			double error = prediction - rating;
			
			rmse = rmse + error * error;
			count = count + 1;
		}
	}

	if(count > 0){
		rmse = sqrt(rmse / count);
	}
	return rmse;
}

// \sigma_{U}^{-2}
double matrixFactorizationEstimateUserFactorPrecision(SocialCovariancePrior *model){
	double userFactorCovariance = 0;

#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for reduction(+:userFactorCovariance)
#endif
	for(int user = 0; user < model -> userCount; user ++){
		for(int k = 0; k < model -> latentFactorCount; k ++){
			double mean = model -> userFactorVariationalMeanMatrix -> entry[user][k];
			double variance = model -> userFactorVariationalVarianceMatrix -> entry[user][k];

			userFactorCovariance = userFactorCovariance + mean * mean + variance;
		}
	}
	userFactorCovariance /= model -> userCount * model -> latentFactorCount;

	double userFactorPrecision = 1.0 / userFactorCovariance;
	return userFactorPrecision;
}

// \sigma_{V}^{-2}
double matrixFactorizationEstimateItemFactorPrecision(SocialCovariancePrior *model){
	double itemFactorCovariance = 0;

#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for reduction(+:itemFactorCovariance)
#endif
	for(int item = 0; item < model -> itemCount; item ++){
		for(int k = 0; k < model -> latentFactorCount; k ++){
			double mean = model -> itemFactorVariationalMeanMatrix -> entry[item][k];
			double variance = model -> itemFactorVariationalVarianceMatrix -> entry[item][k];

			itemFactorCovariance = itemFactorCovariance + mean * mean + variance;
		}
	}
	itemFactorCovariance /= model -> itemCount * model -> latentFactorCount;

	double itemFactorPrecision = 1.0 / itemFactorCovariance;
	return itemFactorPrecision;
}

// \Lambda_{U}^{-1} and \Lambda{Uif} for implicit user social networks
void matrixFactorizationEstimateUserSimilarityMatrix(SocialCovariancePrior *model, Matrix *userSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *userFactorSimilarityRateVector){

	matrixSetValue(userSimilaritySumMatrix, 0);
	matrixSetValue(userSimilarityWeightedSumMatrix, 0);

	int userSamplingCount = (model -> similaritySamplingCount < model -> userCount - 1)
		? model -> similaritySamplingCount: model -> userCount - 1;
	Matrix userFactorSimilarityMeanVector;
	matrixInitialize(&userFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&userFactorSimilarityMeanVector, 0);
	for(int user = 0; user < model -> userCount; user ++){
		for(int f = 0; f < userSamplingCount; f ++){
			int friend;
			do{
				friend = randomSampleInteger(0, model -> userCount - 1);
			}while(friend == user);

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double similarity = 1.0;
				if(model -> userFriendSimilarityClosed == false){
					double userMean = model -> userFactorVariationalMeanMatrix -> entry[user][k];
					double friendMean = model -> userFactorVariationalMeanMatrix -> entry[friend][k];
					double userVariance = model -> userFactorVariationalVarianceMatrix -> entry[user][k];
					double friendVariance = model -> userFactorVariationalVarianceMatrix -> entry[friend][k];

					double similarityRate = ((userMean - friendMean) * (userMean - friendMean)
							+ userVariance + friendVariance + userFactorSimilarityRateVector -> entry[0][k]);
					similarity = 1.0 / similarityRate;
				}
				double normalizedSimilarity = (model -> latentFactorCount + 1) * similarity / userSamplingCount;
			
				userSimilaritySumMatrix -> entry[user][k] += normalizedSimilarity;
				userSimilaritySumMatrix -> entry[friend][k] += normalizedSimilarity;
				
				userSimilarityWeightedSumMatrix -> entry[user][k] += 
					normalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entry[friend][k];
				userSimilarityWeightedSumMatrix -> entry[friend][k] += 
					normalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entry[user][k];

				userFactorSimilarityMeanVector.entry[0][k] += normalizedSimilarity;
			}	
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		userFactorSimilarityMeanVector.entry[0][k] /= model -> userCount;
		userFactorSimilarityRateVector -> entry[0][k] = model -> latentFactorCount / userFactorSimilarityMeanVector.entry[0][k];
	}
	matrixReleaseSpace(&userFactorSimilarityMeanVector);
}

// \Lambda_{V}^{-1} and \Lambda{Vjg} for implicit item social networks
void matrixFactorizationEstimateItemSimilarityMatrix(SocialCovariancePrior *model, Matrix *itemSimilaritySumMatrix,
		Matrix *itemSimilarityWeightedSumMatrix, Matrix *itemFactorSimilarityRateVector){

	matrixSetValue(itemSimilaritySumMatrix, 0);
	matrixSetValue(itemSimilarityWeightedSumMatrix, 0);

	int itemSamplingCount = (model -> similaritySamplingCount < model -> itemCount - 1)
		? model -> similaritySamplingCount: model -> itemCount - 1;
	Matrix itemFactorSimilarityMeanVector;
	matrixInitialize(&itemFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&itemFactorSimilarityMeanVector, 0);
	for(int item = 0; item < model -> itemCount; item ++){
		for(int f = 0; f < itemSamplingCount; f ++){
			int friend;
			do{
				friend = randomSampleInteger(0, model -> itemCount - 1);
			}while(friend == item);

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double similarity = 1.0;
				if(model -> itemFriendSimilarityClosed == false){
					double itemMean = model -> itemFactorVariationalMeanMatrix -> entry[item][k];
					double friendMean = model -> itemFactorVariationalMeanMatrix -> entry[friend][k];
					double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entry[item][k];
					double friendVariance = model -> itemFactorVariationalVarianceMatrix -> entry[friend][k];
					
					double similarityRate = ((itemMean - friendMean) * (itemMean - friendMean)
							+ itemVariance + friendVariance + itemFactorSimilarityRateVector -> entry[0][k]);
					similarity = 1.0 / similarityRate;
				}
				double normalizedSimilarity = (model -> latentFactorCount + 1) * similarity / itemSamplingCount;

				itemSimilaritySumMatrix -> entry[item][k] += normalizedSimilarity;
				itemSimilaritySumMatrix -> entry[friend][k] += normalizedSimilarity;
				
				itemSimilarityWeightedSumMatrix -> entry[item][k] += 
					normalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entry[friend][k];
				itemSimilarityWeightedSumMatrix -> entry[friend][k] += 
					normalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entry[item][k];

				itemFactorSimilarityMeanVector.entry[0][k] += normalizedSimilarity;
			}
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		itemFactorSimilarityMeanVector.entry[0][k] /= model -> itemCount;
		itemFactorSimilarityRateVector -> entry[0][k] = model -> latentFactorCount / itemFactorSimilarityMeanVector.entry[0][k];
	}
	matrixReleaseSpace(&itemFactorSimilarityMeanVector);
}

// \Lambda_{U}^{-1}, \Lambda{Uif} for explicit user social networks
void matrixFactorizationEstimateUserSparseSimilarityMatrix(SocialCovariancePrior *model, Matrix *userSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *userFactorSimilarityRateVector, List *userFriendList){

	matrixSetValue(userSimilaritySumMatrix, 0);
	matrixSetValue(userSimilarityWeightedSumMatrix, 0);

	Matrix userFactorSimilarityMeanVector;
	matrixInitialize(&userFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&userFactorSimilarityMeanVector, 0);
	for(int user = 0; user < model -> userCount; user ++){
		for(int f = 0; f < userFriendList -> columnCounts[user]; f ++){
			int friend = userFriendList -> entry[user][f].key;

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double similarity = 1.0;
				if(model -> userFriendSimilarityClosed == false){
					double userMean = model -> userFactorVariationalMeanMatrix -> entry[user][k];
					double friendMean = model -> userFactorVariationalMeanMatrix -> entry[friend][k];
					double userVariance = model -> userFactorVariationalVarianceMatrix -> entry[user][k];
					double friendVariance = model -> userFactorVariationalVarianceMatrix -> entry[friend][k];

					double similarityRate = ((userMean - friendMean) * (userMean - friendMean)
							+ userVariance + friendVariance + userFactorSimilarityRateVector -> entry[0][k]);
					similarity = 1.0 / similarityRate;
				}
				double userNormalizedSimilarity = (model -> latentFactorCount + 1) * similarity / userFriendList -> columnCounts[user];
				userSimilaritySumMatrix -> entry[user][k] += userNormalizedSimilarity;
				userSimilarityWeightedSumMatrix -> entry[user][k] += 
					userNormalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entry[friend][k];

				if(userFriendList -> columnCounts[friend] > 0){
					double friendNormalizedSimilarity = similarity / userFriendList -> columnCounts[friend];
					userSimilaritySumMatrix -> entry[friend][k] += friendNormalizedSimilarity;
					userSimilarityWeightedSumMatrix -> entry[friend][k] += 
						friendNormalizedSimilarity * model -> userFactorVariationalMeanMatrix -> entry[user][k];
				}

				userFactorSimilarityMeanVector.entry[0][k] += userNormalizedSimilarity;
			}
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		userFactorSimilarityMeanVector.entry[0][k] /= model -> userCount;
		userFactorSimilarityRateVector -> entry[0][k] = model -> latentFactorCount / userFactorSimilarityMeanVector.entry[0][k];
	}
	matrixReleaseSpace(&userFactorSimilarityMeanVector);
}

// \Lambda_{V}^{-1}, \Lambda{Vjg} for explicit item social networks
void matrixFactorizationEstimateItemSparseSimilarityMatrix(SocialCovariancePrior *model, Matrix *itemSimilaritySumMatrix,
		Matrix *itemSimilarityWeightedSumMatrix, Matrix *itemFactorSimilarityRateVector, List *itemFriendList){

	matrixSetValue(itemSimilaritySumMatrix, 0);
	matrixSetValue(itemSimilarityWeightedSumMatrix, 0);

	Matrix itemFactorSimilarityMeanVector;
	matrixInitialize(&itemFactorSimilarityMeanVector, 1, model -> latentFactorCount);
	matrixSetValue(&itemFactorSimilarityMeanVector, 0);
	for(int item = 0; item < model -> itemCount; item ++){
		for(int f = 0; f < itemFriendList -> columnCounts[item]; f ++){
			int friend = itemFriendList -> entry[item][f].key;
			
			for(int k = 0; k < model -> latentFactorCount; k ++){
				double similarity = 1.0;
				if(model -> itemFriendSimilarityClosed == false){
					double itemMean = model -> itemFactorVariationalMeanMatrix -> entry[item][k];
					double friendMean = model -> itemFactorVariationalMeanMatrix -> entry[friend][k];
					double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entry[item][k];
					double friendVariance = model -> itemFactorVariationalVarianceMatrix -> entry[friend][k];

					double similarityRate = ((itemMean - friendMean) * (itemMean - friendMean)
							+ itemVariance + friendVariance + itemFactorSimilarityRateVector -> entry[0][k]);
					similarity = 1.0 / similarityRate;
				}
				double itemNormalizedSimilarity = (model -> latentFactorCount + 1) * similarity / itemFriendList -> columnCounts[item];
				itemSimilaritySumMatrix -> entry[item][k] += itemNormalizedSimilarity;	
				itemSimilarityWeightedSumMatrix -> entry[item][k] += 
					itemNormalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entry[friend][k];

				if(itemFriendList -> columnCounts[friend] > 0){
					double friendNormalizedSimilarity = similarity / itemFriendList -> columnCounts[friend];
					itemSimilaritySumMatrix -> entry[friend][k] += friendNormalizedSimilarity;
					itemSimilarityWeightedSumMatrix -> entry[friend][k] += 
						friendNormalizedSimilarity * model -> itemFactorVariationalMeanMatrix -> entry[item][k];
				}

				itemFactorSimilarityMeanVector.entry[0][k] += itemNormalizedSimilarity;
			}	
		}
	}
	for(int k = 0; k < model -> latentFactorCount; k ++){
		itemFactorSimilarityMeanVector.entry[0][k] /= model -> itemCount;
		itemFactorSimilarityRateVector -> entry[0][k] = model -> latentFactorCount / itemFactorSimilarityMeanVector.entry[0][k];
	}
	matrixReleaseSpace(&itemFactorSimilarityMeanVector);
}

// \sigma_{R}^{-2}
double matrixFactorizationEstimateRatingPrecision(SocialCovariancePrior *model, List *ratingList){
	double ratingVariance = 0.0;
	int ratingCount = listCountEntries(ratingList);

#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for reduction(+:ratingVariance)
#endif
	for(int user = 0; user < ratingList -> rowCount; user ++){
		for(int j = 0; j < ratingList -> columnCounts[user]; j ++){
			int item = ratingList -> entry[user][j].key;
			double rating = ratingList -> entry[user][j].value;

			double variationalMeanDotProduct = 0.0;
			double variationalVarianceDotProduct = 0.0;
			double variationalUserMeanSquareItemVarianceDotProduct = 0.0;
			double variationalItemMeanSquareUserVarianceDotProduct = 0.0;
			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userMean = model -> userFactorVariationalMeanMatrix -> entry[user][k];
				double itemMean = model -> itemFactorVariationalMeanMatrix -> entry[item][k];
				double userVariance = model -> userFactorVariationalVarianceMatrix -> entry[user][k];
				double itemVariance = model -> itemFactorVariationalVarianceMatrix -> entry[item][k];

				variationalMeanDotProduct += userMean * itemMean;

				variationalVarianceDotProduct += userVariance * itemVariance;

				variationalUserMeanSquareItemVarianceDotProduct += userMean * userMean * itemVariance;
				
				variationalItemMeanSquareUserVarianceDotProduct += itemMean * itemMean * userVariance;
			}

			ratingVariance = ratingVariance + rating * rating
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

// \lambda_{Ui} and \gamma_{Ui}
void matrixFactorizationEstimateUserFactorVariationalMeanVarianceMatrices(SocialCovariancePrior *model, List *userRatingList,
		double userFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *userSimilarityWeightedSumMatrix,
		double ratingPrecision){
	
#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for
#endif
	for(int user = 0; user < model -> userCount; user ++){
		Matrix momentMatrix;
		matrixInitialize(&momentMatrix, model -> latentFactorCount, model -> latentFactorCount);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			for(int l = 0; l <= k; l ++){
				double moment = 0.0;
				for(int j = 0; j < userRatingList -> columnCounts[user]; j ++){
					int item = userRatingList -> entry[user][j].key;

					moment += model -> itemFactorVariationalMeanMatrix -> entry[item][k] * model -> itemFactorVariationalMeanMatrix -> entry[item][l];
					if(k == l){
						moment += model -> itemFactorVariationalVarianceMatrix -> entry[item][k];
					}
				}
				moment *= ratingPrecision;
				
				if(k == l){
					moment += userFactorPrecision * model -> userFactorRegularizationRatio;
				}
				
				if(k == l){
					moment += userSimilaritySumMatrix -> entry[user][k] * model -> userFriendRegularizationRatio;
				}
				
				momentMatrix.entry[k][l] = moment;
				momentMatrix.entry[l][k] = moment;
			}
		}

		Matrix momentInverseMatrix;
		matrixInitialize(&momentInverseMatrix, model -> latentFactorCount, model -> latentFactorCount);
		matrixGetInverse(&momentMatrix, &momentInverseMatrix);
		matrixReleaseSpace(&momentMatrix);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			model -> userFactorVariationalVarianceMatrix -> entry[user][k] = momentInverseMatrix.entry[k][k];
		}
		// =================================
		
		Matrix weightedRatingVector;
		matrixInitialize(&weightedRatingVector, 1, model -> latentFactorCount);

		for(int k = 0; k < model -> latentFactorCount; k ++){
			double weightedRating = 0.0;
			for(int j = 0; j < userRatingList -> columnCounts[user]; j ++){
				int item = userRatingList -> entry[user][j].key;
				double rating = userRatingList -> entry[user][j].value;

				weightedRating += rating * model -> itemFactorVariationalMeanMatrix -> entry[item][k];
			}
			weightedRating *= ratingPrecision;

			weightedRating += userSimilarityWeightedSumMatrix -> entry[user][k] * model -> userFriendRegularizationRatio;

			weightedRatingVector.entry[0][k] = weightedRating;
		}
		
		// =================================

		for(int k = 0; k  < model -> latentFactorCount; k ++){
			model -> userFactorVariationalMeanMatrix -> entry[user][k] = vectorCalculateDotProduct(
					momentInverseMatrix.entry[k], weightedRatingVector.entry[0], model -> latentFactorCount);
		}

		matrixReleaseSpace(&momentInverseMatrix);
		matrixReleaseSpace(&weightedRatingVector);
	}
}

// \lambda_{Vj} and \gamma_{Vj}
void matrixFactorizationEstimateItemFactorVariationalMeanVarianceMatrices(SocialCovariancePrior *model, List *itemRatingList,
		double itemFactorPrecision,
		Matrix *itemSimilaritySumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		double ratingPrecision){

#ifdef SCP_PARALLEL_OPEN
	#pragma omp parallel for
#endif
	for(int item = 0; item < model -> itemCount; item ++){
		Matrix momentMatrix;
		matrixInitialize(&momentMatrix, model -> latentFactorCount, model -> latentFactorCount);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			for(int l = 0; l <= k; l ++){
				double moment = 0.0;
				for(int j = 0; j < itemRatingList -> columnCounts[item]; j ++){
					int user = itemRatingList -> entry[item][j].key;

					moment += model -> userFactorVariationalMeanMatrix -> entry[user][k] * model -> userFactorVariationalMeanMatrix -> entry[user][l];
					if(k == l){
						moment += model -> userFactorVariationalVarianceMatrix -> entry[user][k];
					}
				}
				moment *= ratingPrecision;
				
				if(k == l){
					moment += itemFactorPrecision * model -> itemFactorRegularizationRatio;
				}
				
				if(k == l){
					moment += itemSimilaritySumMatrix -> entry[item][k] * model -> itemFriendRegularizationRatio;
				}
				
				momentMatrix.entry[k][l] = moment;
				momentMatrix.entry[l][k] = moment;
			}
		}

		Matrix momentInverseMatrix;
		matrixInitialize(&momentInverseMatrix, model -> latentFactorCount, model -> latentFactorCount);
		matrixGetInverse(&momentMatrix, &momentInverseMatrix);
		matrixReleaseSpace(&momentMatrix);
		
		for(int k = 0; k < model -> latentFactorCount; k ++){
			model -> itemFactorVariationalVarianceMatrix -> entry[item][k] = momentInverseMatrix.entry[k][k];
		}
		// =================================
		
		Matrix weightedRatingVector;
		matrixInitialize(&weightedRatingVector, 1, model -> latentFactorCount);

		for(int k = 0; k < model -> latentFactorCount; k ++){
			double weightedRating = 0.0;
			for(int i = 0; i < itemRatingList -> columnCounts[item]; i ++){
				int user = itemRatingList -> entry[item][i].key;
				double rating = itemRatingList -> entry[item][i].value;

				weightedRating += rating * model -> userFactorVariationalMeanMatrix -> entry[user][k];
			}
			weightedRating *= ratingPrecision;
			
			weightedRating += itemSimilarityWeightedSumMatrix -> entry[item][k] * model -> itemFriendRegularizationRatio;
			
			weightedRatingVector.entry[0][k] = weightedRating;
		}
		
		// =================================

		for(int k = 0; k  < model -> latentFactorCount; k ++){
			model -> itemFactorVariationalMeanMatrix -> entry[item][k] = vectorCalculateDotProduct(
					momentInverseMatrix.entry[k], weightedRatingVector.entry[0], model -> latentFactorCount);
		}

		matrixReleaseSpace(&momentInverseMatrix);
		matrixReleaseSpace(&weightedRatingVector);

	}
}

void matrixFactorizationRunVMStep(SocialCovariancePrior *model, List *ratingList,
		double *userFactorPrecision, double *itemFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *itemSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		Matrix *userFactorSimilarityRateVector, Matrix *itemFactorSimilarityRateVector,
		double *ratingPrecision,
		List *userFriendList, List *itemFriendList){
	
	*ratingPrecision = matrixFactorizationEstimateRatingPrecision(model, ratingList);
	
	if(model -> userFactorRegularizationRatio > 0){
		*userFactorPrecision = matrixFactorizationEstimateUserFactorPrecision(model);
	}

	if(model -> itemFactorRegularizationRatio > 0){
		*itemFactorPrecision = matrixFactorizationEstimateItemFactorPrecision(model);
	}
	
	if(model -> userFriendRegularizationRatio > 0){
		if(model -> userFriendImported){
			matrixFactorizationEstimateUserSparseSimilarityMatrix(model, userSimilaritySumMatrix, 
					userSimilarityWeightedSumMatrix, userFactorSimilarityRateVector, userFriendList);
		}else{
			matrixFactorizationEstimateUserSimilarityMatrix(model, userSimilaritySumMatrix, 
					userSimilarityWeightedSumMatrix, userFactorSimilarityRateVector);
		}
	}

	if(model -> itemFriendRegularizationRatio > 0){
		if(model -> itemFriendImported){
			matrixFactorizationEstimateItemSparseSimilarityMatrix(model, itemSimilaritySumMatrix, 
					itemSimilarityWeightedSumMatrix, itemFactorSimilarityRateVector, itemFriendList);
		}else{
			matrixFactorizationEstimateItemSimilarityMatrix(model, itemSimilaritySumMatrix, 
					itemSimilarityWeightedSumMatrix, itemFactorSimilarityRateVector);
		}
	}
}

void matrixFactorizationRunVEStep(SocialCovariancePrior *model, List *userRatingList, List *itemRatingList,
		double userFactorPrecision, double itemFactorPrecision,
		Matrix *userSimilaritySumMatrix, Matrix *itemSimilaritySumMatrix,
		Matrix *userSimilarityWeightedSumMatrix, Matrix *itemSimilarityWeightedSumMatrix,
		double ratingPrecision){

	matrixFactorizationEstimateUserFactorVariationalMeanVarianceMatrices(model, userRatingList,
		userFactorPrecision, 
		userSimilaritySumMatrix, userSimilarityWeightedSumMatrix,
		ratingPrecision);

	matrixFactorizationEstimateItemFactorVariationalMeanVarianceMatrices(model, itemRatingList, 
		itemFactorPrecision,
		itemSimilaritySumMatrix, itemSimilarityWeightedSumMatrix,
		ratingPrecision);
}

void matrixFactorizationLearn(SocialCovariancePrior *model, List *trainingRatingList, List *validationRatingList,
		List *userFriendList, List *itemFriendList, char *ratingFilePath){

	List *userRatingList = trainingRatingList;
	listSortRows(userRatingList);
	
	List itemRatingList;
	listInitialize(&itemRatingList, model -> itemCount);
	listGetReverseList(trainingRatingList, &itemRatingList);

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
	matrixSetRandomValues(&userSimilaritySumMatrix, 0, 1);
	matrixSetRandomValues(&itemSimilaritySumMatrix, 0, 1);
	
	Matrix userSimilarityWeightedSumMatrix;
	Matrix itemSimilarityWeightedSumMatrix;	
	matrixInitialize(&userSimilarityWeightedSumMatrix, model -> userCount, model -> latentFactorCount);		// N * K
	matrixInitialize(&itemSimilarityWeightedSumMatrix, model -> itemCount, model -> latentFactorCount);		// M * K
	matrixSetRandomValues(&userSimilarityWeightedSumMatrix, 0, 1);
	matrixSetRandomValues(&itemSimilarityWeightedSumMatrix, 0, 1);
	
	matrixSetRandomValues(model -> userFactorVariationalMeanMatrix, 0, 1);
	matrixSetRandomValues(model -> itemFactorVariationalMeanMatrix, 0, 1);
	matrixSetRandomValues(model -> userFactorVariationalVarianceMatrix, 0, 1);
	matrixSetRandomValues(model -> itemFactorVariationalVarianceMatrix, 0, 1);
	
	Optimizer optimizer;
	optimizerInitialize(&optimizer);
	
	double lastValidationCost = DBL_MAX;
	for(int epoch = 0; epoch < optimizer.epochCount; epoch ++){	
		// E step other random variables
		matrixFactorizationRunVEStep(model, userRatingList, &itemRatingList,
			userFactorPrecision, itemFactorPrecision,
			&userSimilaritySumMatrix, &itemSimilaritySumMatrix,
			&userSimilarityWeightedSumMatrix, &itemSimilarityWeightedSumMatrix,
			ratingPrecision);

		// M-step and E-step similarity random variables
		matrixFactorizationRunVMStep(model, trainingRatingList,
			&userFactorPrecision, &itemFactorPrecision,
			&userSimilaritySumMatrix, &itemSimilaritySumMatrix,
			&userSimilarityWeightedSumMatrix, &itemSimilarityWeightedSumMatrix,
			&userFactorSimilarityRateVector, &itemFactorSimilarityRateVector,
			&ratingPrecision,
			userFriendList, itemFriendList);

		double validationCost = matrixFactorizationEvaluateRMSE(model, validationRatingList);
		
		printf("%s\tEpoch %4d\tValidCost %f\tCostDescent %f\t\n", ratingFilePath, epoch + 1,
				validationCost, (epoch > 0) ? lastValidationCost - validationCost : 0.0);

		if(epoch > 0 && lastValidationCost - validationCost < optimizer.convergenceThreshold || isnan(validationCost)){
			printf("%s\tEarly stopping\n", ratingFilePath);
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
	
	listReleaseSpace(&itemRatingList);
}

int main(int argc, char *argv[]){
	srand(time(NULL));

	char *ratingFilePath = argv[1];
	char *userFactorFilePath = argv[2];
	char *itemFactorFilePath = argv[3];
	int latentFactorCount = atoi(argv[4]);	
	double userFriendRatio, itemFriendRatio;
	sscanf(argv[5], "%lf", &userFriendRatio);
	sscanf(argv[6], "%lf", &itemFriendRatio);
	int userFriendImported = atoi(argv[7]);
	int itemFriendImported = atoi(argv[8]);
	char *userFriendFilePath = argv[9];
	char *itemFriendFilePath = argv[10];
	
	// Read rating data
	int userCount, itemCount;
	ratingReadHeader(ratingFilePath, &userCount, &itemCount);
	
	List ratingList;
	listInitialize(&ratingList, userCount);
	ratingReadList(ratingFilePath, &ratingList);
	ratingNormalizeListByMean(&ratingList);
	printf("%d users, %d items\n", userCount, itemCount);

	printf("Split training set and validation set\n");
	double validationRatio = 0.1;
	List trainingRatingList, validationRatingList;
	listInitialize(&trainingRatingList, userCount);
	listInitialize(&validationRatingList, userCount);
	ratingSplitListValidation(&ratingList, &trainingRatingList, &validationRatingList, validationRatio);

	listReleaseSpace(&ratingList);

	// Read social network data
	List userFriendList;
	listInitialize(&userFriendList, userCount);
	if((userFriendImported & 1) > 0){
		ratingReadList(userFriendFilePath, &userFriendList);
	}
	printf("User social network imported: %d\n", userFriendImported);

	List itemFriendList;
	listInitialize(&itemFriendList, itemCount);
	if((itemFriendImported & 1) > 0){
		ratingReadList(itemFriendFilePath, &itemFriendList);
	}
	printf("Item social network imported: %d\n", itemFriendImported);
	
	// Set model
	SocialCovariancePrior model = {
		.latentFactorCount = latentFactorCount,
		.similaritySamplingCount = 100,
		.userCount = userCount,
		.itemCount = itemCount,
		.userFactorRegularizationRatio = 1 - userFriendRatio,
		.itemFactorRegularizationRatio = 1 - itemFriendRatio,
		.userFriendRegularizationRatio = userFriendRatio,
		.itemFriendRegularizationRatio = itemFriendRatio,
		.userFriendImported = (userFriendImported & 1) > 0,
		.itemFriendImported = (itemFriendImported & 1) > 0,
		.userFriendSimilarityClosed = (userFriendImported & 2) > 0,
		.itemFriendSimilarityClosed = (itemFriendImported & 2) > 0
	};

	printf("User friend similarity closed == %s\n", (model.userFriendSimilarityClosed)? "True": "False");
	printf("Item friend similarity closed == %s\n", (model.itemFriendSimilarityClosed)? "True": "False");
	
	Matrix userFactorVariationalMeanMatrix, itemFactorVariationalMeanMatrix;
	matrixInitialize(&userFactorVariationalMeanMatrix, userCount, model.latentFactorCount);
	matrixInitialize(&itemFactorVariationalMeanMatrix, itemCount, model.latentFactorCount);
	model.userFactorVariationalMeanMatrix = &userFactorVariationalMeanMatrix;
	model.itemFactorVariationalMeanMatrix = &itemFactorVariationalMeanMatrix;
	printf("User variational mean matrix: %d %d\n", userFactorVariationalMeanMatrix.rowCount, userFactorVariationalMeanMatrix.columnCount);
	printf("Item variational mean matrix: %d %d\n", itemFactorVariationalMeanMatrix.rowCount, itemFactorVariationalMeanMatrix.columnCount);

	Matrix userFactorVariationalVarianceMatrix, itemFactorVariationalVarianceMatrix;
	matrixInitialize(&userFactorVariationalVarianceMatrix, userCount, model.latentFactorCount);
	matrixInitialize(&itemFactorVariationalVarianceMatrix, itemCount, model.latentFactorCount);
	model.userFactorVariationalVarianceMatrix = &userFactorVariationalVarianceMatrix;
	model.itemFactorVariationalVarianceMatrix = &itemFactorVariationalVarianceMatrix;
	printf("User variational mean matrix: %d %d\n", userFactorVariationalVarianceMatrix.rowCount, userFactorVariationalVarianceMatrix.columnCount);
	printf("Item variational mean matrix: %d %d\n", itemFactorVariationalVarianceMatrix.rowCount, itemFactorVariationalVarianceMatrix.columnCount);
	
	matrixFactorizationLearn(&model, &trainingRatingList, &validationRatingList, &userFriendList, &itemFriendList, ratingFilePath);
	
	ratingWriteMatrix(userFactorFilePath, model.userFactorVariationalMeanMatrix);
	ratingWriteMatrix(itemFactorFilePath, model.itemFactorVariationalMeanMatrix);

	// Release space
	listReleaseSpace(&trainingRatingList);
	listReleaseSpace(&validationRatingList);
	matrixReleaseSpace(&userFactorVariationalMeanMatrix);
	matrixReleaseSpace(&itemFactorVariationalMeanMatrix);
	matrixReleaseSpace(&userFactorVariationalVarianceMatrix);
	matrixReleaseSpace(&itemFactorVariationalVarianceMatrix);
	listReleaseSpace(&userFriendList);
	listReleaseSpace(&itemFriendList);

	printf("%s\tOK\n", ratingFilePath);
	return 0;
}
