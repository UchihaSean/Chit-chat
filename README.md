# Chit-chat

### Seq2seq (sub model for Reinforcement learning)
> - python 2.7
> - tensorflow 1.8.0

### Files
> - Data Preprocessing ------- data_preprocess.py data/data_ch.py
> - Seq2seq model ------------ seq2seq_model.py
> - Train inference ---------- train.py
> - Pred inference ----------- seq2seq.py

### Data preprcessing
> - Filter origin data with the sequence length 
> - Save word to vector and vector to word

### Model seq2seq
_Details in_ https://www.tensorflow.org/versions/r1.2/tutorials/seq2seq
> - Use tensorflow seq2seq model
> - Feed encoder input, decoder input and targets output to train the model with weights ('<pad>' with 0 weights)
> - For test, just feed encoder input and feed_previous == True (use previous target output as new decoder input)

*plus: Data not uploaded beacuse of the privacy*