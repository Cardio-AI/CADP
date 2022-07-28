# Content-Aware Differential Privacy with Conditional Invertible Neural Networks

Malte Tölle, Ullrich Köthe, Florian André, Benjamin Meder, and Sandy Engelhardt

Code for our paper accepted at the 3rd MICCAI workshop on Distributed, Collaborative and Federated Learning (DeCaF) 2022.

Paper link: [https://arxiv.org/](https://arxiv.org/)

## Abstract

Differential privacy (DP) has arisen as the gold standard in protecting an individual's privacy in datasets by adding calibrated noise to each data sample. 
While the application to categorical data is straightforward, its usability in the context of images has been limited. 
Contrary to categorical data the meaning of an image is inherent in the spatial correlation of neighboring pixels making the simple application of noise infeasible.
Invertible Neural Networks (INN) have shown excellent generative performance while still providing the ability to quantify the exact likelihood. % of the data.
Their principle is based on transforming a complicated distribution into a simple one e.g.\ an image into a spherical Gaussian.
We hypothesize that adding noise to the latent space of an INN can enable differentially private image modification.
Manipulation of the latent space leads to a modified image while preserving important details.
Further, by conditioning the INN on meta-data provided with the dataset we aim at leaving dimensions important for downstream tasks like classification untouched while altering other parts that potentially contain identifying information.
We term our method \textit{content-aware differential privacy} (CADP).
We conduct experiments on publicly available benchmarking datasets as well as dedicated medical ones.
In addition, we show the generalizability of our method to categorical data.

## BibTeX

```
in work
```

## Contact

Malte Tölle  
[malte.toelle@med.uni-heidelberg.de](mailto:malte.toelle@med.uni-heidelberg.de)  
[@maltetoelle](https://twitter.com/maltetoelle)

Group Artificial Intelligence in Cardiovascular Medicine (AICM) 
Heidelberg University Hospital
Im Neuenheimer Feld 410, 69120 Heidelberg, Germany