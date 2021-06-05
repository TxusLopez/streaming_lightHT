# streaming_lightHT

"Hyperparameter tuning (or optimization) is often treated as a manual task where experienced users define a subset of hyperparameters and their corresponding range of possible values to be tested exhaustively (Grid Search), randomly (Random Search) or according to some other criteria. The brute force approach of trying all possible combinations of hyperparameters and their values is time-consuming but can be efficiently executed in parallel in a batch setting. However, it can be difficult to emulate this approach in an evolving streaming scenario. A naive approach is to separate an initial set of instances from the first instances seen and perform an offline tuning of the model hyperparameters on them. Nevertheless, this makes a strong assumption that even if the concept drifts the selected hyperparameters’ values will remain optimal. The challenge is to design an approach that incorporate the hyperparameter tuning as part of the continual learning process, which might involve data preprocessing, drift detection, drift recovery, and others."[1]. 

The aim of this work is to show how two simple and lightweight approaches (NEIGHBOURS and DIRECTIONS) are competitive with the well-known Successive Halving algorithm and its random version. The results show this comptitiveness in terms of classification performance, being our approaches less processing time-consuming and less memory-consuming (RAM-Hours). 

[1] Gomes, H. M., Read, J., Bifet, A., Barddal, J. P., & Gama, J. (2019). Machine learning for streaming data: state of the art, challenges, and opportunities. ACM SIGKDD Explorations Newsletter, 21(2), 6-22.

## Experiments


## Results

### Window size=50 and short grid

 Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.772+-0.014 | 0.798+-0.035 | 0.776+-0.022 | 0.793+-0.032
| Processing Time | 6.7+-0.1 | 6.7+-0.1 | 1.4+-0.1 | 1.4+-0.1
| RAM-Hours | 8.9e-04+-1.4e-05 | 0.0e+00+-0.0e+00 | 5.3e-05+-4.0e-06 | 6.0e-06+-0.0e+00

Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.699+-0.029 | 0.646+-0.097 | 0.592+-0.054 | 0.626+-0.084
| Processing Time | 6.9+-0.1 | 7.0+-0.1 | 1.5+-0.0 | 1.6+-0.0
| RAM-Hours | 1.1e-04+-1.0e-06 | 1.0e-06+-0.0e+00 | 6.9e-05+-3.0e-06 | 0.0e+00+-0.0e+00

Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.684+-0.069 | 0.753+-0.071 | 0.706+-0.069 | 0.724+-0.096
| Processing Time | 6.9+-0.4 | 7.1+-0.4 | 1.4+-0.0 | 1.4+-0.0
| RAM-Hours | 3.5e-04+-3.4e-05 | 1.6e-04+-1.6e-05 | 2.5e-05+-1.0e-06 | 4.0e-06+-0.0e+00

Dataset:  agrawal_3_4
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.768+-0.011 | 0.769+-0.013 | 0.767+-0.012 | 0.771+-0.015
| Processing Time | 6.5+-0.0 | 6.6+-0.0 | 1.4+-0.0 | 1.4+-0.0
| RAM-Hours | 1.3e-04+-1.0e-06 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_4_5
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.732+-0.013 | 0.747+-0.048 | 0.734+-0.014 | 0.759+-0.034
| Processing Time | 7.2+-0.3 | 7.3+-0.3 | 1.6+-0.1 | 1.7+-0.1
| RAM-Hours | 0.0e+00+-0.0e+00 | 3.0e-05+-3.0e-06 | 1.4e-04+-6.0e-06 | 3.0e-06+-0.0e+00

Dataset:  agrawal_5_6
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.773+-0.015 | 0.781+-0.029 | 0.779+-0.024 | 0.780+-0.024
| Processing Time | 6.6+-0.1 | 6.7+-0.1 | 1.5+-0.1 | 1.6+-0.1
| RAM-Hours | 3.3e-04+-4.0e-06 | 1.0e-06+-0.0e+00 | 8.5e-05+-6.0e-06 | 5.0e-06+-0.0e+00

Dataset:  agrawal_6_7
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.895+-0.008 | 0.895+-0.010 | 0.895+-0.009 | 0.897+-0.010
| Processing Time | 6.1+-0.3 | 6.2+-0.3 | 1.1+-0.0 | 1.2+-0.0
| RAM-Hours | 2.7e-04+-2.2e-05 | 0.0e+00+-0.0e+00 | 2.3e-05+-1.0e-06 | 0.0e+00+-0.0e+00

Dataset:  agrawal_7_8
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.001 | 0.884+-0.013 | 0.908+-0.004 | 0.901+-0.009
| Processing Time | 6.7+-0.1 | 6.8+-0.1 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.1e-04+-3.0e-06 | 3.0e-05+-1.0e-06 | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_8_9
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.907+-0.005 | 0.907+-0.006 | 0.907+-0.006 | 0.907+-0.006
| Processing Time | 4.9+-0.0 | 5.0+-0.0 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 3.8e-05+-0.0e+00 | 5.0e-06+-0.0e+00 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  mixed
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.861+-0.025 | 0.835+-0.064 | 0.834+-0.062 | 0.839+-0.053
| Processing Time | 4.7+-0.1 | 4.7+-0.1 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 5.0e-05+-1.0e-06 | 0.0e+00+-0.0e+00 | 7.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  randomRBF
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.802+-0.039 | 0.798+-0.044 | 0.801+-0.039 | 0.793+-0.048
| Processing Time | 7.0+-0.1 | 7.1+-0.0 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.5e-04+-1.0e-06 | 2.0e-06+-0.0e+00 | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sea_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.907+-0.009 | 0.901+-0.022 | 0.905+-0.012 | 0.890+-0.025
| Processing Time | 4.5+-0.1 | 4.6+-0.1 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 6.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 8.0e-06+-0.0e+00 | 4.0e-06+-0.0e+00

Dataset:  sea_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.018 | 0.900+-0.025 | 0.901+-0.024 | 0.901+-0.022
| Processing Time | 4.6+-0.1 | 4.7+-0.1 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 9.4e-05+-4.0e-06 | 5.2e-05+-2.0e-06 | 3.4e-05+-1.0e-06 | 2.0e-06+-0.0e+00

Dataset:  sea_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.914+-0.009 | 0.908+-0.016 | 0.895+-0.037 | 0.901+-0.024
| Processing Time | 4.6+-0.0 | 4.6+-0.0 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 1.8e-04+-1.0e-06 | 1.3e-05+-0.0e+00 | 5.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00

Dataset:  stagger_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.859+-0.116 | 0.859+-0.117 | 0.860+-0.117 | 0.855+-0.121
| Processing Time | 5.8+-0.1 | 5.9+-0.1 | 1.2+-0.0 | 1.3+-0.0
| RAM-Hours | 4.4e-05+-1.0e-06 | 2.0e-06+-0.0e+00 | 7.0e-05+-4.0e-06 | 1.0e-06+-0.0e+00

Dataset:  stagger_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.963+-0.005 | 0.963+-0.005 | 0.958+-0.015 | 0.932+-0.047
| Processing Time | 5.6+-0.1 | 5.7+-0.1 | 1.1+-0.1 | 1.2+-0.1
| RAM-Hours | 3.2e-05+-1.0e-06 | 2.1e-05+-1.0e-06 | 1.1e-05+-1.0e-06 | 0.0e+00+-0.0e+00

Dataset:  sine_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.864+-0.019 | 0.888+-0.046 | 0.871+-0.043 | 0.900+-0.036
| Processing Time | 3.8+-0.1 | 3.9+-0.1 | 0.9+-0.1 | 1.0+-0.1
| RAM-Hours | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.835+-0.015 | 0.881+-0.036 | 0.876+-0.024 | 0.866+-0.027
| Processing Time | 3.8+-0.0 | 3.8+-0.0 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 8.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.852+-0.018 | 0.845+-0.056 | 0.848+-0.035 | 0.855+-0.031
| Processing Time | 3.7+-0.0 | 3.7+-0.0 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 1.3e-05+-0.0e+00 | 0.0e+00+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  image_segments
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.701+-0.012 | 0.689+-0.039 | 0.682+-0.012 | 0.663+-0.032
| Processing Time | 8.1+-5.2 | 8.3+-5.3 | 4.2+-0.6 | 4.4+-0.6
| RAM-Hours | 1.3e-03+-1.1e-03 | 2.5e-05+-3.1e-05 | 6.9e-05+-1.8e-05 | 3.0e-06+-1.0e-06

Dataset:  phising
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.748+-0.011 | 0.757+-0.033 | 0.779+-0.008 | 0.774+-0.022
| Processing Time | 3.1+-2.8 | 3.2+-2.8 | 1.5+-0.0 | 1.5+-0.0
| RAM-Hours | 1.7e-05+-1.9e-05 | 3.0e-06+-7.0e-06 | 4.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

**Mean results:**
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08
| Processing Time | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08
| RAM-Hours | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08

### Window size=300 and short grid

 Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.760+-0.014 | 0.764+-0.018 | 0.745+-0.052 | 0.703+-0.056
| Processing Time | 43.7+-0.1 | 44.1+-0.1 | 9.6+-0.2 | 10.0+-0.2
| RAM-Hours | 1.1e-02+-1.0e-05 | 2.2e-03+-5.0e-06 | 1.8e-04+-5.0e-06 | 1.0e-06+-0.0e+00

Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.702+-0.033 | 0.703+-0.037 | 0.628+-0.083 | 0.669+-0.078
| Processing Time | 45.5+-0.7 | 45.8+-0.7 | 9.6+-0.2 | 10.0+-0.2
| RAM-Hours | 3.4e-03+-1.1e-04 | 3.7e-04+-1.1e-05 | 8.7e-04+-1.8e-05 | 3.6e-04+-1.3e-05

Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.721+-0.044 | 0.725+-0.052 | 0.718+-0.056 | 0.708+-0.061
| Processing Time | 44.6+-1.4 | 44.9+-1.4 | 9.8+-0.6 | 10.2+-0.6
| RAM-Hours | 1.4e-03+-5.5e-05 | 1.6e-04+-8.0e-06 | 2.4e-03+-2.0e-04 | 3.0e-06+-0.0e+00

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

**Mean results:**

| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 3.1e-03$\pm$9.3e-04 | 3.3e-04$\pm$1.7e-04 | 2.9e-04$\pm$2.0e-05 | 7.8e-05$\pm$4.1e-06
| Processing Time | 3.1e-03$\pm$9.3e-04 | 3.3e-04$\pm$1.7e-04 | 2.9e-04$\pm$2.0e-05 | 7.8e-05$\pm$4.1e-06
| RAM-Hours | 3.1e-03$\pm$9.3e-04 | 3.3e-04$\pm$1.7e-04 | 2.9e-04$\pm$2.0e-05 | 7.8e-05$\pm$4.1e-06

### Window size=30 and large grid


 Dataset:  agrawal_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.765+-0.014 | 0.777+-0.040 | 0.787+-0.035 | 0.779+-0.034
| Processing Time | 148.7+-1.9 | 148.8+-1.9 | 1.3+-0.1 | 1.4+-0.1
| RAM-Hours | 7.9e-01+-2.3e-04 | 3.8e-04+-9.0e-06 | 4.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.691+-0.005 | 0.664+-0.046 | 0.639+-0.039 | 0.668+-0.017
| Processing Time | 156.7+-4.8 | 156.8+-4.8 | 1.5+-0.1 | 1.6+-0.1
| RAM-Hours | 7.3e-01+-4.9e-04 | 1.2e-03+-7.6e-05 | 1.0e-06+-0.0e+00 | 7.0e-06+-1.0e-06

Dataset:  agrawal_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.683+-0.032 | 0.728+-0.049 | 0.736+-0.047 | 0.738+-0.049
| Processing Time | 151.4+-3.0 | 151.5+-3.0 | 1.4+-0.1 | 1.4+-0.1
| RAM-Hours | 7.5e-01+-9.4e-04 | 7.6e-04+-2.1e-05 | 4.6e-05+-8.0e-06 | 0.0e+00+-0.0e+00

Dataset:  agrawal_3_4
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.762+-0.014 | 0.766+-0.020 | 0.766+-0.022 | 0.768+-0.024
| Processing Time | 144.9+-0.7 | 145.0+-0.7 | 1.3+-0.1 | 1.4+-0.1
| RAM-Hours | 7.0e-01+-1.0e-04 | 9.0e-05+-1.0e-06 | 1.1e-04+-1.6e-05 | 0.0e+00+-0.0e+00

Dataset:  agrawal_4_5
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.736+-0.007 | 0.745+-0.039 | 0.756+-0.020 | 0.765+-0.020
| Processing Time | 161.9+-1.0 | 162.0+-1.0 | 1.6+-0.1 | 1.6+-0.1
| RAM-Hours | 6.6e-01+-1.4e-04 | 9.0e-04+-9.0e-06 | 6.6e-05+-5.0e-06 | 1.5e-05+-1.0e-06

Dataset:  agrawal_5_6
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.791+-0.014 | 0.802+-0.018 | 0.799+-0.023 | 0.801+-0.021
| Processing Time | 147.0+-3.8 | 147.1+-3.8 | 1.5+-0.1 | 1.5+-0.1
| RAM-Hours | 6.6e-01+-4.2e-04 | 3.6e-04+-1.8e-05 | 8.0e-06+-0.0e+00 | 3.0e-06+-0.0e+00

Dataset:  agrawal_6_7
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.900+-0.010 | 0.902+-0.011 | 0.901+-0.012 | 0.901+-0.012
| Processing Time | 132.3+-0.8 | 132.4+-0.8 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 4.3e-01+-4.7e-05 | 8.9e-05+-1.0e-06 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_7_8
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.903+-0.002 | 0.895+-0.016 | 0.892+-0.014 | 0.896+-0.010
| Processing Time | 146.2+-0.9 | 146.3+-0.9 | 1.5+-0.1 | 1.5+-0.1
| RAM-Hours | 4.8e-01+-8.3e-05 | 5.0e-03+-6.4e-05 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  agrawal_8_9
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.910+-0.003 | 0.909+-0.004 | 0.910+-0.003 | 0.910+-0.004
| Processing Time | 112.3+-0.8 | 112.4+-0.8 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 2.8e-01+-4.3e-05 | 4.1e-04+-6.0e-06 | 0.0e+00+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  mixed
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.871+-0.015 | 0.871+-0.021 | 0.871+-0.015 | 0.861+-0.029
| Processing Time | 102.1+-0.6 | 102.2+-0.6 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 1.2e-01+-1.9e-05 | 2.9e-05+-0.0e+00 | 1.1e-05+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  randomRBF
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.829+-0.025 | 0.822+-0.030 | 0.819+-0.033 | 0.822+-0.031
| Processing Time | 151.3+-0.9 | 151.5+-0.9 | 1.4+-0.0 | 1.5+-0.0
| RAM-Hours | 5.4e-01+-7.7e-05 | 5.1e-05+-1.0e-06 | 2.0e-06+-0.0e+00 | 1.0e-06+-0.0e+00

Dataset:  sea_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.913+-0.004 | 0.909+-0.009 | 0.899+-0.020 | 0.897+-0.022
| Processing Time | 99.5+-1.1 | 99.6+-1.1 | 1.0+-0.0 | 1.0+-0.0
| RAM-Hours | 1.9e-02+-2.0e-04 | 3.7e-03+-8.1e-05 | 1.1e-05+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sea_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.917+-0.005 | 0.915+-0.010 | 0.910+-0.013 | 0.912+-0.012
| Processing Time | 99.9+-0.5 | 100.0+-0.5 | 1.0+-0.0 | 1.1+-0.0
| RAM-Hours | 1.0e-02+-4.8e-05 | 3.0e-06+-0.0e+00 | 5.8e-05+-3.0e-06 | 1.0e-06+-0.0e+00

Dataset:  sea_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.920+-0.005 | 0.902+-0.022 | 0.917+-0.012 | 0.911+-0.010
| Processing Time | 100.0+-0.3 | 100.0+-0.3 | 1.0+-0.1 | 1.1+-0.1
| RAM-Hours | 4.1e-03+-1.3e-05 | 7.6e-05+-0.0e+00 | 9.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  stagger_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.915+-0.021 | 0.910+-0.024 | 0.910+-0.027 | 0.907+-0.025
| Processing Time | 127.6+-0.7 | 127.7+-0.7 | 1.2+-0.0 | 1.2+-0.0
| RAM-Hours | 4.1e-02+-5.5e-05 | 5.9e-04+-5.0e-06 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  stagger_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.964+-0.005 | 0.964+-0.005 | 0.954+-0.020 | 0.934+-0.036
| Processing Time | 122.1+-0.5 | 122.2+-0.5 | 1.1+-0.0 | 1.1+-0.0
| RAM-Hours | 1.8e-02+-2.6e-05 | 1.0e-06+-0.0e+00 | 5.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_0_1
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.869+-0.015 | 0.914+-0.018 | 0.889+-0.038 | 0.905+-0.031
| Processing Time | 83.8+-0.4 | 83.9+-0.4 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 3.3e-03+-1.3e-05 | 2.6e-04+-2.0e-06 | 1.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_1_2
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.841+-0.001 | 0.893+-0.020 | 0.881+-0.021 | 0.876+-0.011
| Processing Time | 82.8+-0.4 | 82.9+-0.4 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 2.0e-03+-1.2e-05 | 4.0e-06+-0.0e+00 | 7.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  sine_2_3
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.846+-0.014 | 0.838+-0.050 | 0.839+-0.037 | 0.850+-0.026
| Processing Time | 80.8+-0.3 | 80.9+-0.3 | 0.9+-0.0 | 0.9+-0.0
| RAM-Hours | 3.4e-03+-2.1e-05 | 8.0e-06+-0.0e+00 | 2.0e-06+-0.0e+00 | 0.0e+00+-0.0e+00

Dataset:  image_segments
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.697+-0.015 | 0.657+-0.069 | 0.666+-0.028 | 0.580+-0.102
| Processing Time | 177.1+-117.7 | 177.2+-117.8 | 4.1+-0.6 | 4.3+-0.6
| RAM-Hours | 7.9e-01+-7.4e-01 | 5.9e-05+-1.1e-04 | 1.1e-04+-4.0e-05 | 2.2e-05+-8.0e-06

Dataset:  phising
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 0.765+-0.020 | 0.717+-0.044 | 0.781+-0.013 | 0.774+-0.020
| Processing Time | 66.6+-61.1 | 66.6+-61.1 | 1.5+-0.1 | 1.5+-0.1
| RAM-Hours | 7.3e-02+-8.6e-02 | 1.7e-04+-3.8e-04 | 3.4e-05+-4.0e-06 | 2.0e-06+-0.0e+00

**Mean results:**

| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 3.4e-01$\pm$3.9e-02 | 6.7e-04$\pm$3.7e-05 | 2.3e-05$\pm$3.6e-06 | 2.4e-06$\pm$4.8e-07
| Processing Time | 3.4e-01$\pm$3.9e-02 | 6.7e-04$\pm$3.7e-05 | 2.3e-05$\pm$3.6e-06 | 2.4e-06$\pm$4.8e-07
| RAM-Hours | 3.4e-01$\pm$3.9e-02 | 6.7e-04$\pm$3.7e-05 | 2.3e-05$\pm$3.6e-06 | 2.4e-06$\pm$4.8e-07

## How to replicate the experiments




**Este texto está en negrita**
*Este texto está en cursiva*
Tal como dice Abraham Lincoln:

> Con perdón de la expresión

- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request
