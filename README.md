# Determining How Citations Are Used in Citation Contexts

We propose the following *types of citations*:

Given a citation context (e.g., sentence), the citation
1. is grammatically integrated in the sentence, 
2. is annotated directly after the occurrence of author names, 
3. backs up a concept,
4. backs up a claim, or
5. is not appropriate because the context is incomplete or noisy.

We argue that determining such classes for citation contexts is useful for a variety of tasks, such as improved citation recommendation and scientific impact quantification.

This repository contains the [training](train-500-sw.xlsx) and [test](test-100-nlp.xlsx) data sets, as well as the a [multi-label gradient boosting classifier](contexttypes.py) as machine-learning-based classifier for determining the classes automatically based on citation contexts.


## Contact & Reference
The system has been designed and implemented by Michael Färber and Ashwath Sampath. In case of questions (e.g., requesting the paper) or feedback feel free to reach out to us:

[Michael Färber](https://sites.google.com/view/michaelfaerber), michael.faerber@kit&#46;edu

If you use our code or would like to referene our work, please cite our paper as follows:
```
@inproceedings{Faerber2019TPDL,
  author    = {Michael F{\"{a}}rber and Ashwath Sampath},
  title     = "{Determining How Citations are Used in Citation Contexts}",
  booktitle = "{Proceedings of the 23rd International Conference on Theory and Practice of Digital Libraries}",
  series    = "{TPDL'19}",
  location  = "{Oslo, Norway}",
  year      = {2019}
}
```
