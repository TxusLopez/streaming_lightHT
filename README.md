# streaming_lightHT

"Hyperparameter tuning (or optimization) is often treated as a manual task where experienced users define a subset of hyperparameters and their corresponding range of possible values to be tested exhaustively (Grid Search), randomly (Random Search) or according to some other criteria. The brute force approach of trying all possible combinations of hyperparameters and their values is time-consuming but can be efficiently executed in parallel in a batch setting. However, it can be difficult to emulate this approach in an evolving streaming scenario. A naive approach is to separate an initial set of instances from the first instances seen and perform an offline tuning of the model hyperparameters on them. Nevertheless, this makes a strong assumption that even if the concept drifts the selected hyperparameters’ values will remain optimal. The challenge is to design an approach that incorporate the hyperparameter tuning as part of the continual learning process, which might involve data preprocessing, drift detection, drift recovery, and others."[1]. 

The aim of this work is to show how two simple and lightweight approaches are competitive with the well-known Successive Halving algorithm in terms of classification performance, being less processing time-consuming and using less memory (RAM-Hours). 

[1] Gomes, H. M., Read, J., Bifet, A., Barddal, J. P., & Gama, J. (2019). Machine learning for streaming data: state of the art, challenges, and opportunities. ACM SIGKDD Explorations Newsletter, 21(2), 6-22.

## Experiments


## Results


## How to replicate the experiments




**Este texto está en negrita**
*Este texto está en cursiva*
Tal como dice Abraham Lincoln:

> Con perdón de la expresión

- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request
