# streaming_lightHT

"Hyperparameter tuning (or optimization) is often treated as a manual task where experienced users define a subset of hyperparameters and their corresponding range of possible values to be tested exhaustively (Grid Search), randomly (Random Search) or according to some other criteria. The brute force approach of trying all possible combinations of hyperparameters and their values is time-consuming but can be efficiently executed in parallel in a batch setting. However, it can be difficult to emulate this approach in an evolving streaming scenario. A naive approach is to separate an initial set of instances from the first instances seen and perform an offline tuning of the model hyperparameters on them. Nevertheless, this makes a strong assumption that even if the concept drifts the selected hyperparameters’ values will remain optimal. The challenge is to design an approach that incorporate the hyperparameter tuning as part of the continual learning process, which might involve data preprocessing, drift detection, drift recovery, and others."[1]. 

The aim of this work is to show how two simple and lightweight approaches are competitive with the well-known Successive Halving algorithm in terms of classification performance, being less processing time-consuming and using less memory (RAM-Hours). 

[1] Gomes, H. M., Read, J., Bifet, A., Barddal, J. P., & Gama, J. (2019). Machine learning for streaming data: state of the art, challenges, and opportunities. ACM SIGKDD Explorations Newsletter, 21(2), 6-22.

## Experiments


## Results


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

**Mean results**
| METRICS | Halving | Random Halving | NEIGHBOURS | DIRECT
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Prequential acc. | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08
| Processing Time | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08
| RAM-Hours | 2.0e-04$\pm$5.9e-05 | 1.7e-05$\pm$2.9e-06 | 3.0e-05$\pm$2.1e-06 | 1.4e-06$\pm$4.8e-08

## How to replicate the experiments




**Este texto está en negrita**
*Este texto está en cursiva*
Tal como dice Abraham Lincoln:

> Con perdón de la expresión

- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request
