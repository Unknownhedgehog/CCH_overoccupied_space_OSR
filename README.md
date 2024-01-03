# Resilience to the flowing unknown: an Open Set Recognition Framework for Data Streams
![alt text](streaming_osr_framework.jpg)
## Abstract
Modern digital applications extensively integrate Artificial Intelligence (AI) models into their core systems, offering significant advantages for automated decision-making. 
However, these AI-based systems encounter reliability and safety challenges when handling continuously generated data streams in complex and dynamic scenarios. 
This work explores the concept of resilient AI systems, which must operate in the face of unexpected events, including instances belonging to patterns that have not been seen during the training process. 
This is an issue that regular closed-set classifiers commonly encounter in streaming scenarios, as they are designed to compulsory classify any new observation into one of the training patterns 
(i.e., the so-called *over-occupied space* problem). In batch learning, the Open Set Recognition (OSR) research area has consistently confronted this issue 
by requiring models to robustly uphold their classification performance when processing query instances from unknown patterns. In this context, 
this work investigates the application of an OSR framework that combines classification and clustering to address the *over-occupied space* problem in streaming scenarios. 
Specifically, we systematically devise a benchmark comprising different classification datasets with varying ratios of known to unknown classes. 
Experiments over this benchmark are presented to compare the performance of the proposed hybrid framework to that of individual incremental classifiers. 
Discussions held over the obtained results highlight situations where the proposed framework performs best, and delineate 
the limitations and hurdles encountered by incremental classifiers in effectively resolving the challenges posed by open-world streaming environments.
## Keywords
resilient AI, open set recognition, unknown classes, streaming, over-occupied space

## Datasets
The files `isogauss_params.py` and `hypercube_params.py` define the parameters that are used for creating the datasets and 
running the experiments. Each list contains the values of a parameter for the creation of each dataset.
* n_samples: number of samples.
* center_classes: number of classes.
* features: number of features.
* cluster_std: intra-cluster standard deviation.
* class_sep: factor multiplying the hypercube size.
* rand_state: random number generation for dataset creation.

The values in each list correspond in order to the dataset number. Each value can be removed, added or changed in order to 
modify the number of datasets and their parameters.


## Experiments and evaluation
The experiments can be run for each type of datasets with the following commands:

IsoGauss datasets experiments:
```console
python isogauss_tests.py
```
HyperCube datasets experiments:
```console
python hypercube_tests.py
```
Insects datasets experiments:
```console
python insects_tests.py
```
After finishing, a .txt file will be generated containing a table with the results of each metric (Acc, K-Acc, U-Acc, 
N-Acc, F1, AUC, DB index) for each approach (*static*, *incremental* and *sOSR*) of the corresponding dataset. Each
generated file also contains 2 additional tables with the *p-values* from a non-parametric Wilcoxon signed-rank test,
performed between the *static* and *sOSR* baselines and the *incremental* and *sOSR* baselines.