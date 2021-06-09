# streaming_lightHT

"Hyperparameter tuning (or optimization) is often treated as a manual task where experienced users define a subset of hyperparameters and their corresponding range of possible values to be tested exhaustively (Grid Search), randomly (Random Search) or according to some other criteria. The brute force approach of trying all possible combinations of hyperparameters and their values is time-consuming but can be efficiently executed in parallel in a batch setting. However, it can be difficult to emulate this approach in an evolving streaming scenario. A naive approach is to separate an initial set of instances from the first instances seen and perform an offline tuning of the model hyperparameters on them. Nevertheless, this makes a strong assumption that even if the concept drifts the selected hyperparameters’ values will remain optimal. The challenge is to design an approach that incorporate the hyperparameter tuning as part of the continual learning process, which might involve data preprocessing, drift detection, drift recovery, and others."[1]. 

The aim of this work is to show how two simple and lightweight approaches (NEIGHBOURS and DIRECTIONS) are competitive with the well-known Successive Halving algorithm (SHA) and its random version (RSHA). The results show this competitiveness in terms of classification performance, being our approaches less processing time-consuming and less memory-consuming (RAM-Hours). 

Three experiments have been carried out. 1) with a sliding window size of 50 and a short grid of parameters, 2) with a sliding window size of 300 and a short grid of parameters, and 3) with a sliding window size of 50 and a large grid of parameters.

[1] Gomes, H. M., Read, J., Bifet, A., Barddal, J. P., & Gama, J. (2019). Machine learning for streaming data: state of the art, challenges, and opportunities. ACM SIGKDD Explorations Newsletter, 21(2), 6-22.

## Data generation

The datasets can be generated with the file "data_gen_v0.py" (see *source_code* folder). In the VARIABLES section of the file, the variable *length_dataset* set the number of instances that will be included in the stream. The variable *change_width* set the width of the drift, being 1 for abrupt drifts, and larger values for more gradual ones. The variable *datas* is a list of all datasets considered for generation. And *path_data* set the place where the datasets (in csv format) will be placed in your computer.

In case you need to generate other datasets, it is very simple. You can modify the function *data_preparation*, and follow the guidelines of the river framework (https://riverml.xyz/latest/).

Under the folder "datasets" you can find the ones used for this research.

## Replicating the experiments

The experiments can be replicated by using the file "icdm2021_v0.py" (see *source_code* folder). In the VARIABLES section of the file, first we find the variables that correspond to the grid for the Hoeffding Tree. Here we can use the short or the large mode. Next we can set the scoring (*scoring*), the testing size for the "train_test_split" process (*tst_size*), and the number of repetitions (*runs*) for the experiment with each dataset. Also the *window_size*. Next we can configure the number of iterations in the search process of NEIGHBOURS and DIRECTIONS (*iterations_neighs*, *iterations_direct*). The parameter for the Successive Halving approaches (SHA, RSHA) are *eta*, *budget*, *n_models_sh_random*, and *budget_random*. Finally, there are the variables for the reading of the data, and the name of the datasets themselves.

Once the process has finishes, we can have the results in pkl files. Under the folder "results" you can find the ones produced by this research in each of the 3 experiments mentioned at the begining.

## Packages and dependencies

We have used the following libraries and packages: pandas, warnings, traceback, collections, timeit, river, numpy, random, pickle, psutil, and itertools.

## Results

### Mean metrics in 25 runs for Window size=50 and short grid:

 #### Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.772+-0.014 | 0.798+-0.035 | 0.776+-0.022 | 0.793+-0.032
| Processing Time | 6.7+-0.1 | 6.7+-0.1 | 1.4+-0.1 | 1.4+-0.1
| RAM-Hours | 8.9e-04+-1.4e-05 | 0.0e+00+-0.0e+00 | 5.3e-05+-4.0e-06 | 6.0e-06+-0.0e+00

![agrawal_0_1_(w50_short)](/results/window_30/images/agrawal_0_1.png)

#### Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.699+-0.029 | 0.646+-0.097 | 0.592+-0.054 | 0.626+-0.084
| Processing Time | 6.9+-0.1 | 7.0+-0.1 | 1.5+-0.0 | 1.6+-0.0
| RAM-Hours | 1.1e-04+-1.0e-06 | 1.0e-06+-0.0e+00 | 6.9e-05+-3.0e-06 | 0.0e+00+-0.0e+00

![agrawal_1_2_(w50_short)](/results/window_30/images/agrawal_1_2.png)

#### Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.684+-0.069 | 0.753+-0.071 | 0.706+-0.069 | 0.724+-0.096
| Processing Time | 6.9+-0.4 | 7.1+-0.4 | 1.4+-0.0 | 1.4+-0.0
| RAM-Hours | 3.5e-04+-3.4e-05 | 1.6e-04+-1.6e-05 | 2.5e-05+-1.0e-06 | 4.0e-06+-0.0e+00

![agrawal_2_3_(w50_short)](/results/window_30/images/agrawal_2_3.png)

#### Dataset:  agrawal_3_4
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.768+-0.011 | 0.769+-0.013 | 0.767+-0.012 | 0.771+-0.015
| Processing Time | 6.5+-0.0 | 6.6+-0.0 | 1.4+-0.0 | 1.4+-0.0
| RAM-Hours | 1.3e-04+-1.0e-06 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

![agrawal_3_4_(w50_short)](/results/window_30/images/agrawal_3_4.png)

#### Dataset:  agrawal_4_5
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.732+-0.013 | 0.747+-0.048 | 0.734+-0.014 | 0.759+-0.034
| Processing Time | 7.2+-0.3 | 7.3+-0.3 | 1.6+-0.1 | 1.7+-0.1
| RAM-Hours | 0.0e+00+-0.0e+00 | 3.0e-05+-3.0e-06 | 1.4e-04+-6.0e-06 | 3.0e-06+-0.0e+00

![agrawal_4_5_(w50_short)](/results/window_30/images/agrawal_4_5.png)

#### Dataset:  agrawal_5_6
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.773+-0.015 | 0.781+-0.029 | 0.779+-0.024 | 0.780+-0.024
| Processing Time | 6.6+-0.1 | 6.7+-0.1 | 1.5+-0.1 | 1.6+-0.1
| RAM-Hours | 3.3e-04+-4.0e-06 | 1.0e-06+-0.0e+00 | 8.5e-05+-6.0e-06 | 5.0e-06+-0.0e+00

![agrawal_5_6_(w50_short)](/results/window_30/images/agrawal_5_6.png)

#### Dataset:  agrawal_6_7
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.895+-0.008 | 0.895+-0.010 | 0.895+-0.009 | 0.897+-0.010
| Processing Time | 6.1+-0.3 | 6.2+-0.3 | 1.1+-0.0 | 1.2+-0.0
| RAM-Hours | 2.7e-04+-2.2e-05 | 0.0e+00+-0.0e+00 | 2.3e-05+-1.0e-06 | 0.0e+00+-0.0e+00

![agrawal_6_7_(w50_short)](/results/window_30/images/agrawal_6_7.png)

#### Dataset:  agrawal_7_8
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.001 | 0.884+-0.013 | 0.908+-0.004 | 0.901+-0.009
| Processing Time | 6.7+-0.1 | 6.8+-0.1 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.1e-04+-3.0e-06 | 3.0e-05+-1.0e-06 | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![agrawal_7_8_(w50_short)](/results/window_30/images/agrawal_7_8.png)

#### Dataset:  agrawal_8_9
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.907+-0.005 | 0.907+-0.006 | 0.907+-0.006 | 0.907+-0.006
| Processing Time | 4.9+-0.0 | 5.0+-0.0 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 3.8e-05+-0.0e+00 | 5.0e-06+-0.0e+00 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![agrawal_8_9_(w50_short)](/results/window_30/images/agrawal_8_9.png)

#### Dataset:  mixed
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.861+-0.025 | 0.835+-0.064 | 0.834+-0.062 | 0.839+-0.053
| Processing Time | 4.7+-0.1 | 4.7+-0.1 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 5.0e-05+-1.0e-06 | 0.0e+00+-0.0e+00 | 7.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![mixed_(w50_short)](/results/window_30/images/mixed.png)

#### Dataset:  randomRBF
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.802+-0.039 | 0.798+-0.044 | 0.801+-0.039 | 0.793+-0.048
| Processing Time | 7.0+-0.1 | 7.1+-0.0 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.5e-04+-1.0e-06 | 2.0e-06+-0.0e+00 | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![randomRBF_(w50_short)](/results/window_30/images/randomRBF.png)

#### Dataset:  sea_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.907+-0.009 | 0.901+-0.022 | 0.905+-0.012 | 0.890+-0.025
| Processing Time | 4.5+-0.1 | 4.6+-0.1 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 8.0e-06+-0.0e+00 | 4.0e-06+-0.0e+00

![sea_0_1_(w50_short)](/results/window_30/images/sea_0_1.png)

#### Dataset:  sea_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.018 | 0.900+-0.025 | 0.901+-0.024 | 0.901+-0.022
| Processing Time | 4.6+-0.1 | 4.7+-0.1 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 9.4e-05+-4.0e-06 | 5.2e-05+-2.0e-06 | 3.4e-05+-1.0e-06 | 2.0e-06+-0.0e+00

![sea_1_2_(w50_short)](/results/window_30/images/sea_1_2.png)

#### Dataset:  sea_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.914+-0.009 | 0.908+-0.016 | 0.895+-0.037 | 0.901+-0.024
| Processing Time | 4.6+-0.0 | 4.6+-0.0 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 1.8e-04+-1.0e-06 | 1.3e-05+-0.0e+00 | 5.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00

![sea_2_3_(w50_short)](/results/window_30/images/sea_2_3.png)

#### Dataset:  stagger_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.859+-0.116 | 0.859+-0.117 | 0.860+-0.117 | 0.855+-0.121
| Processing Time | 5.8+-0.1 | 5.9+-0.1 | 1.2+-0.0 | 1.3+-0.0
| RAM-Hours | 4.4e-05+-1.0e-06 | 2.0e-06+-0.0e+00 | 7.0e-05+-4.0e-06 | 1.0e-06+-0.0e+00

![stagger_0_1_(w50_short)](/results/window_30/images/stagger_0_1.png)

#### Dataset:  stagger_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.963+-0.005 | 0.963+-0.005 | 0.958+-0.015 | 0.932+-0.047
| Processing Time | 5.6+-0.1 | 5.7+-0.1 | 1.1+-0.1 | 1.2+-0.1
| RAM-Hours | 3.2e-05+-1.0e-06 | 2.1e-05+-1.0e-06 | 1.1e-05+-1.0e-06 | 0.0e+00+-0.0e+00

![stagger_1_2_(w50_short)](/results/window_30/images/stagger_1_2.png)

#### Dataset:  sine_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.864+-0.019 | 0.888+-0.046 | 0.871+-0.043 | 0.900+-0.036
| Processing Time | 3.8+-0.1 | 3.9+-0.1 | 0.9+-0.1 | 1.0+-0.1
| RAM-Hours | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![sine_0_1_(w50_short)](/results/window_30/images/sine_0_1.png)

#### Dataset:  sine_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.835+-0.015 | 0.881+-0.036 | 0.876+-0.024 | 0.866+-0.027
| Processing Time | 3.8+-0.0 | 3.8+-0.0 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 8.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![sine_1_2_(w50_short)](/results/window_30/images/sine_1_2.png)

#### Dataset:  sine_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.852+-0.018 | 0.845+-0.056 | 0.848+-0.035 | 0.855+-0.031
| Processing Time | 3.7+-0.0 | 3.7+-0.0 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 1.3e-05+-0.0e+00 | 0.0e+00+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![sine_2_3_(w50_short)](/results/window_30/images/sine_2_3.png)

#### Dataset:  image_segments
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.701+-0.012 | 0.689+-0.039 | 0.682+-0.012 | 0.663+-0.032
| Processing Time | 8.1+-5.2 | 8.3+-5.3 | 4.2+-0.6 | 4.4+-0.6
| RAM-Hours | 1.3e-03+-1.1e-03 | 2.5e-05+-3.1e-05 | 6.9e-05+-1.8e-05 | 3.0e-06+-1.0e-06

![image_segments_(w50_short)](/results/window_30/images/image_segments.png)

#### Dataset:  phising
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.748+-0.011 | 0.757+-0.033 | 0.779+-0.008 | 0.774+-0.022
| Processing Time | 3.1+-2.8 | 3.2+-2.8 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.7e-05+-1.9e-05 | 3.0e-06+-7.0e-06 | 4.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

![phising_(w50_short)](/results/window_30/images/phising.png)

**Mean results for all datasets:**
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | $0.826\pm0.022$ | $0.829\pm0.039$ | $0.823\pm0.031$ | $0.825\pm0.038$
| Processing Time | $5.6\pm0.5$ | $5.7\pm0.5$ | $1.4\pm0.1$ | $1.4\pm0.1$
| RAM-Hours | $2.0e-04\pm5.9e-05$ | $1.7e-05\pm2.9e-06$ | $3.0e-05\pm2.1e-06$ | $1.4e-06\pm4.8e-08$

**Nemenyi test:**

For Prequential acc.:
![Nemenyi test](/results/window_30/images/pACC_nemenyi.png)

For Processing time:
![Nemenyi test](/results/window_30/images/pT_nemenyi.png)

For RAM-Hours:
![Nemenyi test](/results/window_30/images/RAM_nemenyi.png)

### Mean metrics in 25 runs for Window size=300 and short grid

#### Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.760+-0.014 | 0.764+-0.018 | 0.745+-0.052 | 0.703+-0.056
| Processing Time | 43.7+-0.1 | 44.1+-0.1 | 9.6+-0.2 | 10.0+-0.2
| RAM-Hours | 1.1e-02+-1.0e-05 | 2.2e-03+-5.0e-06 | 1.8e-04+-5.0e-06 | 1.0e-06+-0.0e+00

![agrawal_0_1_(w300_short)](/results/window_300/images/agrawal_0_1.png)

#### Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.702+-0.033 | 0.703+-0.037 | 0.628+-0.083 | 0.669+-0.078
| Processing Time | 45.5+-0.7 | 45.8+-0.7 | 9.6+-0.2 | 10.0+-0.2
| RAM-Hours | 3.4e-03+-1.1e-04 | 3.7e-04+-1.1e-05 | 8.7e-04+-1.8e-05 | 3.6e-04+-1.3e-05

![agrawal_1_2_(w300_short)](/results/window_300/images/agrawal_1_2.png)

#### Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.721+-0.044 | 0.725+-0.052 | 0.718+-0.056 | 0.708+-0.061
| Processing Time | 44.6+-1.4 | 44.9+-1.4 | 9.8+-0.6 | 10.2+-0.6
| RAM-Hours | 1.4e-03+-5.5e-05 | 1.6e-04+-8.0e-06 | 2.4e-03+-2.0e-04 | 3.0e-06+-0.0e+00

![agrawal_2_3_(w300_short)](/results/window_300/images/agrawal_2_3.png)

Dataset:  agrawal_3_4
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.713+-0.015 | 0.717+-0.021 | 0.711+-0.016 | 0.716+-0.025
| Processing Time | 45.5+-0.3 | 45.8+-0.3 | 8.8+-0.3 | 9.1+-0.4
| RAM-Hours | 2.4e-03+-1.5e-05 | 0.0e+00+-0.0e+00 | 5.6e-05+-3.0e-06 | 0.0e+00+-0.0e+00

Dataset:  agrawal_4_5
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.697+-0.003 | 0.688+-0.021 | 0.740+-0.039 | 0.717+-0.040
| Processing Time | 41.2+-1.2 | 41.5+-1.2 | 9.8+-0.2 | 10.2+-0.3
| RAM-Hours | 2.5e-04+-1.0e-05 | 6.0e-06+-0.0e+00 | 2.8e-04+-3.0e-06 | 1.4e-04+-3.0e-06

Dataset:  agrawal_5_6
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.739+-0.009 | 0.726+-0.016 | 0.726+-0.017 | 0.725+-0.015
| Processing Time | 42.2+-1.9 | 42.4+-1.9 | 8.6+-0.6 | 8.9+-0.6
| RAM-Hours | 2.9e-03+-1.5e-04 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00 | 1.0e-06+-0.0e+00

Dataset:  agrawal_6_7
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.829+-0.005 | 0.830+-0.007 | 0.829+-0.006 | 0.829+-0.006
| Processing Time | 33.7+-1.4 | 33.9+-1.4 | 6.9+-0.2 | 7.2+-0.2
| RAM-Hours | 4.4e-03+-3.4e-04 | 4.7e-05+-2.0e-06 | 5.3e-04+-1.5e-05 | 7.4e-05+-3.0e-06

Dataset:  agrawal_7_8
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.821+-0.001 | 0.817+-0.007 | 0.813+-0.010 | 0.819+-0.009
| Processing Time | 41.4+-0.9 | 41.7+-0.9 | 8.6+-0.3 | 9.0+-0.3
| RAM-Hours | 5.8e-03+-2.3e-04 | 5.9e-05+-2.0e-06 | 1.1e-04+-5.0e-06 | 2.9e-05+-2.0e-06

Dataset:  agrawal_8_9
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.830+-0.004 | 0.830+-0.005 | 0.830+-0.005 | 0.829+-0.005
| Processing Time | 30.3+-0.6 | 30.6+-0.7 | 7.0+-0.3 | 7.3+-0.3
| RAM-Hours | 1.7e-03+-7.1e-05 | 1.9e-04+-6.0e-06 | 3.1e-04+-1.7e-05 | 2.5e-05+-1.0e-06

Dataset:  mixed
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.779+-0.039 | 0.751+-0.054 | 0.784+-0.043 | 0.742+-0.074
| Processing Time | 29.0+-1.3 | 29.2+-1.3 | 6.5+-0.4 | 6.8+-0.5
| RAM-Hours | 2.1e-03+-1.1e-04 | 5.6e-05+-5.0e-06 | 1.9e-05+-3.0e-06 | 2.0e-05+-2.0e-06

Dataset:  randomRBF
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.745+-0.023 | 0.730+-0.036 | 0.734+-0.036 | 0.719+-0.052
| Processing Time | 46.5+-1.0 | 46.9+-1.0 | 9.2+-0.2 | 9.5+-0.1
| RAM-Hours | 8.5e-03+-1.9e-04 | 4.5e-04+-1.9e-05 | 9.9e-05+-2.0e-06 | 0.0e+00+-0.0e+00

Dataset:  sea_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.833+-0.004 | 0.832+-0.005 | 0.823+-0.019 | 0.833+-0.008
| Processing Time | 28.2+-0.8 | 28.4+-0.8 | 6.2+-0.4 | 6.5+-0.4
| RAM-Hours | 1.4e-03+-8.1e-05 | 4.8e-04+-2.7e-05 | 2.6e-04+-2.6e-05 | 3.8e-05+-4.0e-06

Dataset:  sea_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.834+-0.003 | 0.833+-0.011 | 0.832+-0.012 | 0.830+-0.010
| Processing Time | 28.3+-0.3 | 28.5+-0.4 | 6.0+-0.2 | 6.3+-0.2
| RAM-Hours | 3.0e-03+-2.6e-05 | 1.7e-04+-3.0e-06 | 4.5e-05+-3.0e-06 | 8.0e-06+-0.0e+00

Dataset:  sea_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.840+-0.006 | 0.840+-0.006 | 0.837+-0.009 | 0.826+-0.027
| Processing Time | 27.9+-0.4 | 28.0+-0.4 | 5.8+-0.1 | 6.1+-0.2
| RAM-Hours | 1.6e-04+-4.0e-06 | 2.0e-04+-6.0e-06 | 6.5e-05+-2.0e-06 | 6.8e-05+-3.0e-06

Dataset:  stagger_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.719+-0.087 | 0.715+-0.095 | 0.719+-0.087 | 0.714+-0.095
| Processing Time | 38.5+-0.1 | 38.7+-0.1 | 7.1+-0.2 | 7.4+-0.2
| RAM-Hours | 5.4e-04+-2.0e-06 | 2.6e-04+-2.0e-06 | 2.6e-05+-2.0e-06 | 1.1e-05+-1.0e-06

Dataset:  stagger_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.882+-0.003 | 0.882+-0.003 | 0.882+-0.003 | 0.877+-0.015
| Processing Time | 33.8+-0.3 | 34.0+-0.3 | 6.5+-0.1 | 6.8+-0.1
| RAM-Hours | 3.8e-04+-4.0e-06 | 1.5e-04+-2.0e-06 | 1.4e-05+-0.0e+00 | 1.0e-05+-0.0e+00

Dataset:  sine_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.778+-0.022 | 0.830+-0.023 | 0.792+-0.044 | 0.826+-0.025
| Processing Time | 22.4+-0.3 | 22.5+-0.3 | 5.3+-0.1 | 5.5+-0.1
| RAM-Hours | 2.6e-04+-5.0e-06 | 1.0e-06+-0.0e+00 | 1.2e-04+-3.0e-06 | 2.1e-04+-8.0e-06

Dataset:  sine_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.775+-0.016 | 0.796+-0.030 | 0.781+-0.029 | 0.778+-0.025
| Processing Time | 23.5+-0.2 | 23.6+-0.2 | 5.2+-0.0 | 5.4+-0.0
| RAM-Hours | 1.9e-04+-3.0e-06 | 3.0e-06+-0.0e+00 | 2.1e-05+-0.0e+00 | 1.0e-06+-0.0e+00

Dataset:  sine_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.790+-0.011 | 0.767+-0.025 | 0.771+-0.012 | 0.763+-0.019
| Processing Time | 23.1+-0.2 | 23.2+-0.2 | 5.4+-0.2 | 5.6+-0.2
| RAM-Hours | 4.8e-04+-4.0e-06 | 1.3e-04+-2.0e-06 | 0.0e+00+-0.0e+00 | 4.0e-06+-0.0e+00

Dataset:  image_segments
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.570+-0.012 | 0.370+-0.127 | 0.571+-0.012 | 0.497+-0.108
| Processing Time | 67.5+-54.9 | 68.1+-55.1 | 32.0+-1.5 | 33.2+-1.6
| RAM-Hours | 1.4e-02+-1.5e-02 | 2.0e-03+-3.4e-03 | 2.4e-04+-2.0e-05 | 6.2e-04+-3.9e-05

Dataset:  phising
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.523+-0.001 | 0.523+-0.001 | 0.526+-0.002 | 0.521+-0.012
| Processing Time | 18.4+-17.6 | 18.6+-17.7 | 8.7+-0.7 | 9.0+-0.7
| RAM-Hours | 1.6e-03+-3.2e-03 | 4.6e-05+-1.6e-04 | 5.0e-04+-9.3e-05 | 3.1e-05+-8.0e-06

**Mean results for all datasets:**

| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | $0.756\pm0.017$ | $0.746\pm0.029$ | $0.752\pm0.028$ | $0.745\pm0.036$
| Processing Time | $36.0\pm4.1$ | $36.2\pm4.1$ | $8.7\pm0.3$ | $9.0\pm0.3$
| RAM-Hours | $3.1e-03\pm9.3e-04$ | $3.3e-04\pm1.7e-04$ | $2.9e-04\pm2.0e-05$ | $7.8e-05\pm4.1e-06$

**Nemenyi test:**

For Prequential acc.:
![Nemenyi test](/results/window_300/images/pACC_nemenyi.png)

For Processing time:
![Nemenyi test](/results/window_300/images/pT_nemenyi.png)

For RAM-Hours:
![Nemenyi test](/results/window_300/images/RAM_nemenyi.png)

### Mean metrics in 25 runs for Window size=50 and large grid

Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.763+-0.013 | 0.769+-0.034 | 0.768+-0.046 | 0.791+-0.033
| Processing Time | 149.8+-1.7 | 149.8+-1.7 | 1.4+-0.1 | 1.4+-0.1
| RAM-Hours | 8.1e-01+-3.6e-04 | 1.6e-04+-3.0e-06 | 4.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.717+-0.019 | 0.691+-0.075 | 0.628+-0.056 | 0.671+-0.040
| Processing Time | 152.2+-4.5 | 152.3+-4.5 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 7.1e-01+-4.9e-04 | 2.0e-05+-1.0e-06 | 1.5e-05+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.712+-0.033 | 0.764+-0.049 | 0.756+-0.066 | 0.763+-0.051
| Processing Time | 152.9+-4.1 | 153.0+-4.1 | 1.4+-0.0 | 1.4+-0.0
| RAM-Hours | 7.7e-01+-6.8e-04 | 9.7e-04+-5.2e-05 | 2.1e-05+-1.0e-06 | 7.0e-06+-0.0e+00

Dataset:  agrawal_3_4
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.768+-0.016 | 0.775+-0.028 | 0.776+-0.027 | 0.777+-0.027
| Processing Time | 147.9+-4.6 | 148.0+-4.6 | 1.4+-0.1 | 1.5+-0.1
| RAM-Hours | 6.9e-01+-6.0e-04 | 3.1e-03+-1.7e-04 | 7.7e-05+-1.0e-05 | 0.0e+00+-0.0e+00

Dataset:  agrawal_4_5
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.733+-0.005 | 0.756+-0.012 | 0.735+-0.022 | 0.760+-0.015
| Processing Time | 148.5+-1.3 | 148.6+-1.3 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 6.0e-01+-2.9e-04 | 0.0e+00+-0.0e+00 | 3.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_5_6
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.784+-0.011 | 0.796+-0.015 | 0.775+-0.016 | 0.790+-0.019
| Processing Time | 142.8+-1.0 | 143.0+-1.0 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 6.5e-01+-1.6e-04 | 6.0e-04+-8.0e-06 | 8.0e-06+-0.0e+00 | 3.1e-05+-2.0e-06

Dataset:  agrawal_6_7
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.908+-0.005 | 0.909+-0.006 | 0.908+-0.006 | 0.908+-0.005
| Processing Time | 131.5+-2.4 | 131.5+-2.4 | 1.1+-0.0 | 1.2+-0.0
| RAM-Hours | 4.2e-01+-2.1e-04 | 1.2e-05+-0.0e+00 | 2.6e-05+-1.0e-06 | 1.0e-06+-0.0e+00

Dataset:  agrawal_7_8
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.001 | 0.895+-0.011 | 0.892+-0.013 | 0.900+-0.007
| Processing Time | 144.1+-0.8 | 144.2+-0.8 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 4.6e-01+-6.2e-05 | 2.4e-04+-2.0e-06 | 3.7e-05+-1.0e-06 | 5.0e-05+-2.0e-06

Dataset:  agrawal_8_9
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.906+-0.009 | 0.905+-0.009 | 0.905+-0.009 | 0.905+-0.010
| Processing Time | 110.6+-0.6 | 110.7+-0.6 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 2.8e-01+-5.0e-05 | 4.6e-04+-5.0e-06 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  mixed
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.865+-0.029 | 0.865+-0.035 | 0.847+-0.061 | 0.843+-0.058
| Processing Time | 100.9+-0.4 | 101.0+-0.4 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 1.2e-01+-3.0e-05 | 9.7e-04+-5.0e-06 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  randomRBF
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.816+-0.021 | 0.806+-0.024 | 0.812+-0.025 | 0.810+-0.025
| Processing Time | 149.0+-1.0 | 149.1+-1.0 | 1.4+-0.0 | 1.5+-0.0
| RAM-Hours | 5.3e-01+-7.9e-05 | 5.2e-03+-6.8e-05 | 4.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sea_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.912+-0.003 | 0.901+-0.022 | 0.905+-0.018 | 0.908+-0.006
| Processing Time | 98.7+-0.5 | 98.8+-0.5 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 4.8e-03+-3.4e-05 | 3.5e-05+-0.0e+00 | 3.9e-05+-1.0e-06 | 3.0e-06+-0.0e+00

Dataset:  sea_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.910+-0.013 | 0.906+-0.021 | 0.904+-0.021 | 0.903+-0.021
| Processing Time | 99.0+-0.6 | 99.0+-0.6 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 2.2e-03+-1.0e-05 | 2.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sea_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.920+-0.006 | 0.899+-0.027 | 0.909+-0.033 | 0.910+-0.013
| Processing Time | 98.9+-0.5 | 99.0+-0.5 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 0.0e+00+-0.0e+00 | 2.0e-06+-0.0e+00 | 1.7e-05+-1.0e-06 | 1.0e-06+-0.0e+00

Dataset:  stagger_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.907+-0.025 | 0.904+-0.028 | 0.904+-0.030 | 0.897+-0.029
| Processing Time | 125.5+-0.5 | 125.6+-0.5 | 1.2+-0.0 | 1.2+-0.0
| RAM-Hours | 2.4e-02+-4.8e-05 | 0.0e+00+-0.0e+00 | 3.8e-05+-1.0e-06 | 0.0e+00+-0.0e+00

Dataset:  stagger_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.966+-0.006 | 0.966+-0.006 | 0.953+-0.028 | 0.938+-0.035
| Processing Time | 123.4+-3.1 | 123.5+-3.1 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 3.1e-02+-5.5e-04 | 3.2e-05+-1.0e-06 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.867+-0.015 | 0.911+-0.018 | 0.886+-0.042 | 0.898+-0.041
| Processing Time | 87.7+-0.7 | 87.8+-0.7 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 1.6e-04+-3.0e-06 | 4.6e-04+-6.0e-06 | 2.1e-05+-1.0e-06 | 9.0e-06+-0.0e+00

Dataset:  sine_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.840+-0.002 | 0.889+-0.019 | 0.875+-0.021 | 0.883+-0.008
| Processing Time | 86.5+-1.2 | 86.5+-1.2 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 4.2e-04+-1.2e-05 | 7.9e-04+-1.3e-05 | 1.4e-05+-1.0e-06 | 1.4e-05+-1.0e-06

Dataset:  sine_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.853+-0.003 | 0.870+-0.011 | 0.846+-0.017 | 0.848+-0.007
| Processing Time | 80.0+-0.6 | 80.0+-0.6 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 1.7e-03+-1.5e-05 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  image_segments
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.705+-0.008 | 0.647+-0.089 | 0.682+-0.017 | 0.647+-0.058
| Processing Time | 172.9+-115.6 | 173.1+-115.7 | 4.0+-0.5 | 4.1+-0.5
| RAM-Hours | 7.6e-01+-7.1e-01 | 2.7e-04+-6.2e-04 | 1.2e-04+-3.0e-05 | 1.9e-05+-5.0e-06

Dataset:  phising
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.775+-0.012 | 0.761+-0.037 | 0.784+-0.005 | 0.781+-0.008
| Processing Time | 64.7+-59.5 | 64.8+-59.5 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 7.2e-02+-8.5e-02 | 1.5e-04+-3.2e-04 | 1.5e-05+-1.0e-06 | 2.0e-06+-0.0e+00

**Mean results for all datasets:**
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | $0.835\pm0.012$ | $0.837\pm0.027$ | $0.831\pm0.028$ | $0.835\pm0.025$
| Processing Time | $122.3\pm9.8$ | $122.3\pm9.8$ | $1.3\pm0.1$ | $1.4\pm0.1$
| RAM-Hours | $3.3e-01\pm3.8e-02$ | $6.4e-04\pm6.1e-05$ | $2.2e-05\pm2.3e-06$ | $6.5e-06\pm4.8e-07$

**Nemenyi test:**

For Prequential acc.:
![Nemenyi test](/results/window_30_large_grid/images/pACC_nemenyi.png)

For Processing time:
![Nemenyi test](/results/window_30_large_grid/images/pT_nemenyi.png)

For RAM-Hours:
![Nemenyi test](/results/window_30_large_grid/images/RAM_nemenyi.png)

