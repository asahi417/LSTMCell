# LSTM (RNN) Cell 
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Recently proposed LSTM (RNN) cells' implementations by tensorflow.
To compare efficacy, language modeling over 
[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42),
 which was downloaded via [PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz),
is conducted.

Available cells:

- [***Highway State Gating***](cells/basic_rnn_cell.py)  
[Ron Shoham and Haim Permuter. "Highway State Gating for Recurrent Highway Networks: improving information flow through time" arxiv 2018](https://arxiv.org/pdf/1805.09238.pdf)
    - Usage  
    ```
    from cells import CustomRNNCell
    cell = CustomRNNCell(highway_state_gate=True, recurrent_highway=True)
    ```
- [***Hyper Networks***](cells/hypernets_cell.py)
[Ha, David, Andrew Dai, and Quoc V. Le. "Hypernetworks." Proceedings of International Conference on Learning Representations (ICLR) 2017.](https://arxiv.org/abs/1609.09106)
    - Usage  
    ```
    from cells import HyperLSTMCell
    cell = HyperLSTMCell()
    ```
- [***Recurrent Highway Network***](cells/basic_rnn_cell.py)
[Zilly, Julian Georg, et al. "Recurrent Highway Networks." International Conference on Machine Learning (ICML) 2017.](https://arxiv.org/abs/1607.03474)
    - Usage  
    ```
    from cells import CustomRNNCell 
    cell = CustomRNNCell(recurrent_highway=True, recurrent_depth=4)
    ```
- [***Key-Value-Predict Attention***](cells/kvp_attention_cell.py)
[Daniluk, Michał, et al. "Frustratingly short attention spans in neural language modeling." Proceedings of International Conference on Learning Representations (ICLR) 2017.](https://arxiv.org/abs/1702.04521)
    - Usage  
    ```
    from cells import KVPAttentionWrapper, CustomLSTMCell
    cell = LSTMCell.CustomLSTMCell()
    attention_layer = KVPAttentionWrapper(cells)
    ```
- [***Vanilla LSTM***](cells/basic_lstm_cell.py)
    - Usage  
    ```
    from cells import LSTMCell
    cell = LSTMCell.CustomLSTMCell()
    ```

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

# Train language model
See the effect of LSTM cells by language modeling:
```
python train.py -m [lstm type] -e [epoch] -t language
```
- lstm type: lstm, rhn, hypernets, kvp, hsg

The middle sized model of [Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural network regularization." arXiv preprint arXiv:1409.2329 (2014)](https://arxiv.org/abs/1409.2329)
is employed as baseline model. Vanilla LSTM achieves better result with two stacked LSTM layers and other LSTM cells 
achieves their best score with single LSTM layer.  

## Brief comparison 
The results on brief experiment over PTB data set is shown by following table.
See [here](hyperparameters) for hyperparameters.

| Cell | Validation perplexity | Epoch | Trainable variables |
| --- | --- | --- | --- |
| vanilla LSTM        | 152.10 | 8  | 19775200 | 
| Hypernets           | 142.12 | 51 | 21537504 |
| KVP attention       | 142.52 | 10 | 21465850 |
| Recurrent Highway   | 138.00 | 15 | 19355300 |
| Highway State Gating| 138.21 | 14 | 19355300 |

Note that this result is by really rough tuning on hyperparameters.

# Other
- This code is supported by python 3 and tensorflow 1.3.0.
