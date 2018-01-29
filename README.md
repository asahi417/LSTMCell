# LSTM (RNN) Cell 
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Recently proposed LSTM (RNN) cell implementation by tensorflow.
Tested by language modeling task for 
[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42).

Following cells are available:

- Hyper Networks
[Ha, David, Andrew Dai, and Quoc V. Le. "Hypernetworks." arXiv preprint arXiv:1609.09106 (2016).](https://arxiv.org/abs/1609.09106)
- Recurrent Highway Network
[Zilly, Julian Georg, et al. "Recurrent highway networks." arXiv preprint arXiv:1607.03474 (2016).](https://arxiv.org/abs/1607.03474)

Each cell utilizes following regularization:

- Variational Dropout (per-sample)
[Gal, Yarin, and Zoubin Ghahramani. "A theoretically grounded application of dropout in recurrent neural networks." Advances in neural information processing systems. 2016.](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
- Recurrent Dropout
[Semeniuta, Stanislau, Aliaksei Severyn, and Erhardt Barth. "Recurrent dropout without memory loss." arXiv preprint arXiv:1603.05118 (2016).](https://arxiv.org/abs/1603.05118)
- Layer Normalization
[Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).](https://arxiv.org/abs/1607.06450)


Data was downloaded via [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).  

# How to use

```
git clone https://github.com/asahi417/LSTMCell
cd LSTMCell
python train.py [target]
```
Setting *target* to *hypernets* or *rhn*, you can learn model based on Hyper Networks or Recurrent Highway Network.  

# Todo
- Neural Architecture Searching
- Layer normalization dosen't improve performance. It could have some bugs...
- Attention cell

# Other
- This code is supported python 3 and tensorflow 1.3.0.