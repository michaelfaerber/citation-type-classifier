# Determining the Linguistic Types of Citations
We propose the following *linguistic types of citations*:
The citation
1. is not needed due to an incomplete or noisy context, 
2. is grammatically integrated in the sentence, 
3. is annotated directly after the occurrence of author names, 
4. backs up a concept, or 
5. backs up a claim.

We argue that determining such classes for citation contexts is useful for a variety of tasks, such as improved citation recommendation and scientific impact quantification.

This repository contains the [training](train-500-sw.xlsx) and [test](test-100-nlp.xlsx) data sets, as well as the a [multi-label gradient boosting classifier](contexttypes.py) as machine-learning-based classifier for determining the classes automatically based on citation contexts.


## Contact
The system has been designed and implemented by Michael Färber and Ashwath Sampath. Feel free to reach out to us:

[Michael Färber](https://sites.google.com/view/michaelfaerber), michael.faerber@cs&#46;kit&#46;edu
