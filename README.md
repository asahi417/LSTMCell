# LSTM (RNN) Cell 
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Recently proposed LSTM (RNN) cell implementation by tensorflow.
Tested by language modeling task for 
[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42).

Following cells are available:

- [***Hyper Networks***](cells/hypernets_cell.py)
[Ha, David, Andrew Dai, and Quoc V. Le. "Hypernetworks." Proceedings of International Conference on Learning Representations (ICLR) 2017.](https://arxiv.org/abs/1609.09106)
- [***Recurrent Highway Network***](cells/basic_rnn_cell.py)
[Zilly, Julian Georg, et al. "Recurrent Highway Networks." International Conference on Machine Learning (ICML) 2017.](https://arxiv.org/abs/1607.03474)
- [***Key-Value-Predict Attention***](cells/kvp_attention_cell.py)
[Daniluk, Michał, et al. "Frustratingly short attention spans in neural language modeling." Proceedings of International Conference on Learning Representations (ICLR) 2017.](https://arxiv.org/abs/1702.04521)

Each cell utilizes following regularization:

- Variational Dropout (per-sample)
[Gal, Yarin, and Zoubin Ghahramani. "A theoretically grounded application of dropout in recurrent neural networks." Advances in neural information processing systems. 2016.](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
- Recurrent Dropout
[Semeniuta, Stanislau, Aliaksei Severyn, and Erhardt Barth. "Recurrent dropout without memory loss." arXiv preprint arXiv:1603.05118 (2016).](https://arxiv.org/abs/1603.05118)
- Layer Normalization
[Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).](https://arxiv.org/abs/1607.06450)


Data was downloaded via [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).  

## Model
To compare the effect of each cell simply, following parameters are fixed:

- epoch: 50
- batch: 20
- sequence step number: 35
- learning rate: 0.5
- learning rate decay: 0.8 (max_epoch: 10)
- gradient max norm: 10 
- keep probability of dropout for state, input: 0.5
- keep probability of dropout for embedding, outputs: 0.75
- hidden unit and embedding dim: 650

These were selected based on the middle sized model of [Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural network regularization." arXiv preprint arXiv:1409.2329 (2014).](https://arxiv.org/abs/1409.2329).
To compare cells fairly, these parameters should be selected by grid searching but
the purpose of this repository is checking the implementation works correctly rather than building wonderful language model,
so the parameters were roughly selected.

The language model is
- **vanilla LSTM** and **Hyper Networks**: 2 layer stacked cell
- **Recurrent Highway**: 1 layer 

Trainable parameter
- LSTM: 19775200
- RHN: 19355300
- Hyper: 21537504
- Attention: 21465850

The result on PTB is summarized as below.

<p align="center">
  <img src="./img/perplexity.png" width="900">
  <br><i> Perplexity for each cell. </i>
</p>

Here, the data is split into *train*, *validation*, and *test* following [Mikolov](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

|Train Perplexity|Cell|
|---|---|
| 26.50 |  vanilla LSTM | 
| 45.80 |  RHN |
| 37.78 |  Hypernets |


| Validation Perplexity | Cell |
|---|---|
| 149.39 | vanilla LSTM | 
| 127.44 |  RHN | 
| 129.94 |  Hypernets | 

| Test Perplexity | Cell |
|---|---|
| 143.31 |  vanilla LSTM | 
| 121.23 |  RHN | 
| 122.49 |  Hypernets | 


# How to use

```
git clone https://github.com/asahi417/LSTMCell
cd LSTMCell
python train.py [target]
```
Setting *target* to *hypernets* or *rhn*, you can learn model based on Hyper Networks or Recurrent Highway Network.
Also you can see the baseline (vanilla LSTM) by 
```
python train.py
```

# Todo
- Neural Architecture Searching
- Layer normalization dose not improve performance. It could have some bugs...
- Weight tying for language model

# Other
- This code is supported python 3 and tensorflow 1.3.0.
