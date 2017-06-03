# Source Code

## Compilation (GCC for example)

```
gcc -std=c11 -O3 -o social_covariance_prior social_covariance_prior.c data_structure.c -lm
```

## Running

```
./social_covariance_prior rating_file K b_U b_V S_U S_V [user_social_network_file] [item_social_network_file]
```

* K: number of latent factors
* b_U: user balance parameter; the value is between 0 and 1
* b_V: item balance parameter; the value is between 0 and 1
* S_U: whether explicit user social network is given or not
  * 1: [user_social_network_file] must be given a file path to import the explicit user social network
  * 0: SCP learns the implicit user social network
* S_V: whether explicit item social network is given or not
  * 1: [item_social_network_file] must be given a file path to import the explicit item social network
  * 0: SCP learns the implicit item social network

## Data format

Rating file:
* 3 values per line to represent a rating record (i, j, r):
  * 1: integer value i (0 <= i <= N - 1): User i 
  * 2: integer value j (0 <= j <= M - 1): Item j
  * 3: floating-point value r: Rating of user i to item j

Explicit user social network file:
* 2 values per line to represent a directed edge (i, f).
  * 1: integer value i (0 <= i <= N - 1): User i
  * 2: integer value f (0 <= f <= N - 1, f != i): User f

Explicit item social network file:
* 2 values per line to represent a directed edge (j, g).
  * 1: integer value j (0 <= j <= M - 1): Item j
  * 2: integer value g (0 <= g <= M - 1, g != j): Item g

## Comments

Since the code is specifically written for our experiments, so the code

1. runs on Linux platforms only. We are not sure if it can run on Windows or other operating systems.
1. runs only the cross validation of training data. No prediction on test data is implemented.
1. does not handle exceptions if it reads invalid data format or invalid argument values.
1. could be lack of readability.
