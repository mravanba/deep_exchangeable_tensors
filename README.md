# Deep Exchangeable Tensor Models

An implementation of the model found in https://arxiv.org/abs/1803.02879. We use deep learning to model interactions across sets of objects, such as user-movie ratings. The canonical representation of such interactions is a matrix with an exchangeability property: the encoding's meaning is not changed by permuting rows or columns. Models should hence be Permutation Equivariant (PE): constrained to make the same predictions across such permutations. We implement a parameter-sharing scheme that could not be made any more expressive without violating PE. 

Requires: 
Python 3, 
Tensorflow, 
MovieLens-100k or MovieLens-1M dataset