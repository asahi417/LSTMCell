# LSTM (RNN) Cell 
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Recently proposed LSTM (RNN) cell implementation by tensorflow.
Tested by language modeling task for 
[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42),
 which was downloaded via [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

Available property:

- [***Highway State Gating***](cells/basic_rnn_cell.py)  
[Ron Shoham and Haim Permuter. "Highway State Gating for Recurrent Highway Networks: improving information flow through time" arxiv 2018](https://arxiv.org/pdf/1805.09238.pdf)
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

The usage is the same as usual LSTM cell of tensorflow.
  

## TODO
- [ ] add sentiment classification example
- [ ] Layer normalization dose not improve performance. Fix it.

# How to use
## setup
```
git clone https://github.com/asahi417/LSTMCell
cd LSTMCell
pip install -r requirements.txt
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xvzf simple-examples.tgz
```

# Train model language model
See the effect of LSTM cells by language modeling:
```
python train.py -m [lstm type] -e [epoch] -t language
```
- lstm type: lstm, rhn, hypernets, kvp, hsg

The middle sized model of [Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural network regularization." arXiv preprint arXiv:1409.2329 (2014)](https://arxiv.org/abs/1409.2329)
is employed as baseline model. 

## Brief comparison 
The results on brief experiment over PTB data set is shown by following table.
See [here](hyperparameters) for hyperparameters.

| Cell | Train perplexity | Validation perplexity | Epoch for the best validation perplexity | Trainable variables |
| --- | --- | --- | --- | --- |
| vanilla LSTM  | 26.50 | 149.39 | 143.31 | 19775200 | 
| Hypernets     | 37.78 | 129.94 | 122.49 | 21537504 |
| KVP attention | 64.56 | 144.65 | 141.11 | 21465850 |
| Recurrent Highway | 45.80 | 127.44 | 121.23 | 19355300 |
| Highway State Gating| 45.80 | 127.44 | 121.23 | 19355300 |


# Other
- This code is supported by python 3 and tensorflow 1.3.0.
