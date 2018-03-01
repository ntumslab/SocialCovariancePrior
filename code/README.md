# Source Code

## Compilation (GCC for example)

```
gcc -std=c11 -O3 -fopenmp -o social_covariance_prior social_covariance_prior.c data_structure.c -lm
```

## Running

```
./social_covariance_prior training_rating_file user_latent_factor_file item_latent_factor_file  K b_U b_V S_U S_V [user_social_network_file] [item_social_network_file]
```

* training_rating_file: input of rating dataset
* user_latent_factor_file: output of vector \lambda_Ui for each user i at line i (0 indexed).
* item_latent_factor_file: output of vector \lambda_Vj for each item j at line j (0 indexed).
* K: number of latent factors
* b_U: user balance parameter; the value is between 0 and 1
* b_V: item balance parameter; the value is between 0 and 1
* S_U: argument on user social network
  * 0: learning an implicit social network where the edge weights are learned in VEM
  * 1: reading an explicit social network where the edge weights are learned in VEM
  * 2: learning an implicit social network where the edge weights are read from the file
  * 3: reading an explicit social network where the edge weights are read from the file
* S_V: argument on item social network
  * 0: learning an implicit social network where the edge weights are learned in VEM
  * 1: reading an explicit social network where the edge weights are learned in VEM
  * 2: learning an implicit social network where the edge weights are read from the file
  * 3: reading an explicit social network where the edge weights are read from the file
* user_social_network_file: input of explicit user social network [ Optional ]
  * The argument can be skipped if S_U = 0 or 2
* item_social_network_file: input of explicit item social network [ Optional ]
  * The argument can be skipped if S_V = 0 or 2

## Data format

Rating file:
* 2 values at the first line to represent (N, M):
  * N: the number of users
  * M: the number of items
* After the first line, 3 values per line to represent a rating record (i, j, r):
  * Integer i (0 <= i <= N - 1): user i 
  * Integer j (0 <= j <= M - 1): item j
  * Floating-point r: rating of user i to item j

Explicit user social network file:
* 2 values at the first line to represent (N, N):
  * N: the number of users.
* After the first line, 3 values per line to represent a directed edge (i, f, [w]).
  * Integer i (0 <= i <= N - 1): user i
  * Integer f (0 <= f <= N - 1, f != i): user f
  * Floating-point w: weight of edge (i, f) [ Optional, default 1 if not given ]

Explicit item social network file:
* 2 values at the first line to represent (M, M):
  * M: the number of items.
* After the first line, 3 values per line to represent a directed edge (j, g, [w]).
  * Integer j (0 <= j <= M - 1): item j
  * Integer g (0 <= g <= M - 1, g != j): item g
  * Floating-point w: weight of edge (j, g) [ Optional, default 1 if not given ]

## Comments

Since the code is specifically written for our experiments on our machine,

1. it is written on Linux platforms. We are not sure if the code can run on Windows or other operating systems.
1. it does not handle exceptions when reading invalid data format or invalid argument values.
1. it could be lack of readability.
1. the version of GCC compiler maybe requires an upgrade for successful compilation in another platform.
