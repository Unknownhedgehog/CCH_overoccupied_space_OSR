# Resilience to the flowing unknown: an Open Set Recognition Framework for Data Streams

## Abstract
Modern digital applications are massively integrating Artificial Intelligence based methods in their core systems. 
While this integration offers inherent advantages for automated decision making, AI-based methods pose challenges related to reliability and safety, particularly when processing continuously generated data streams within complex, dynamically changing scenarios. 
Resilient AI systems are expected to operate in the face of unexpected events. 
Such events include the arrival of instances from patterns that the AI model has not seen previously. 
This is an issue that regular closed-set classifiers commonly face in streaming environments, because they are forced to categorize any new observation into one of the training classes. 
This, leads to the so-called *over-occupied space* problem, which becomes complex to tackle especially when dealing with non-stationary data streams. 
In stationary settings, the Open Set Recognition (OSR) field consistently confronts this issue while requiring models to uphold their classification performance. 
A very scarcely explored OSR approach to deal with the *over-occupied space* problem is based on hybridizing classification and clustering techniques, where the latter can help to overcome such a limitation. 
We explore this OSR approach for real-time (streaming) scenarios, showing the limitations of a single classification model and showcasing how a combination of clustering and classification techniques can be applied to the OSR problem. 
We empirically compare the performance of this hybrid approach to that of single incremental classifiers, highlighting the conditions under which it surpasses them, while also delineating its shortcomings in effectively addressing the *over-occupied space* problem undergone by online classifiers.
## Keywords
resilient AI, open set recognition, unknown classes, streaming, over-occupied space

## Datasets
The files `blobs_params.py` and `class_params.py` define the parameters that are used for creating the datasets and 
running the experiments.
## Experiments and evaluation
Blobs datasets experiments:
```console
python blobs_tests.py
```
Class datasets experiments:
```console
python class_tests.py
```
Insects datasets experiments:
```console
python insects_tests.py
```