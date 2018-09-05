# Deep Models of Interactions Accross Sets

An implementation of the model found in https://arxiv.org/abs/1803.02879. We use deep learning to model interactions across sets of objects, such as user-movie ratings. The canonical representation of such interactions is a matrix with an exchangeability property: the encoding's meaning is not changed by permuting rows or columns. Models should hence be Permutation Equivariant (PE): constrained to make the same predictions across such permutations. We implement a parameter-sharing scheme that could not be made any more expressive without violating PE. 



## Prerequisites: 

Python 3

Tensorflow

MovieLens-100k or MovieLens-1M dataset: retrieved from https://grouplens.org/datasets/movielens/



## Models:

We include sparse implementations of two distinct models (see paper linked above for details):

### Sparse self-supervised model

run with:

```
aaa
```

### Sarse Factorized Autoencoder

sss

