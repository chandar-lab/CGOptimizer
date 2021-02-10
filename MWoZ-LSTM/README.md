###Dataset

Download [dataset](https://github.com/budzianowski/multiwoz) and follow the steps in the repo to construct ``data/input_lang.index2word.json, data/input_lang.word2index.json', data/output_lang.index2word.json, data/output_lang.word2index.json, data/train_dials.json, and data/test_dials.json and data/val_dials.json``

###Train and Test

Then, run ``train.py`` to train Bi-LSTM encoder with Attention decoder model on multiwoz dataset.

To test, run ``test.py`` to test on the model checkpoints with validation and test sets.
