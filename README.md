# Language modeling using LSTM
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

## Training data
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
```
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
```

## Install prerequisite packages
From folder /PATH/LSTMLanguageModeling/, run in Terminal
```
./setup.sh
```

## Train and test the model
From folder /PATH/LSTMLanguageModeling/. run in Terminal
```
python lstm_langmodel.py
```

