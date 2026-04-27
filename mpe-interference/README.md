# mpe-inference
This is a repository to run inference with pretrained mpe-models.

## Which model?
The model architecture is based on the DRCNN:M as described in [this paper](https://ieeexplore.ieee.org/document/9865174). The only difference is that instead of predicting the center frame it predicts a sequence of same length as the input sequence.

It is trained on:
- Schubert Winterreise Dataset
- Beethoven String Quartet Dataset
- Beethoven Piano Sonata Dataset 

## How to use it for prediction?
Check `example.ipynb`