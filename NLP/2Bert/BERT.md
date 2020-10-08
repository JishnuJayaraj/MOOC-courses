# BERT

## Model: sentence sentiment classification

### steps

![img](https://jalammar.github.io/images/distilBERT/sentiment-classifier-1.png)

![img](https://jalammar.github.io/images/distilBERT/distilbert-bert-sentiment-classifier.png)

![img](https://jalammar.github.io/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png)

![img](https://jalammar.github.io/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png)

![img](https://jalammar.github.io/images/distilBERT/bert-training-logistic-regression.png)

### How it works

![img](https://jalammar.github.io/images/distilBERT/bert-distilbert-tokenization-2-token-ids.png)

![img](https://jalammar.github.io/images/distilBERT/bert-model-input-output-1.png)

![img](https://jalammar.github.io/images/distilBERT/bert-model-calssification-output-vector-cls.png)



#### real pipeline

2000 comments with labels

![img](https://jalammar.github.io/images/distilBERT/sst2-df-head.png)

![img](https://jalammar.github.io/images/distilBERT/sst2-text-to-tokenized-ids-bert-example.png)

padding to make all sentence sequal size 66

![img](https://jalammar.github.io/images/distilBERT/bert-input-tensor.png)

run model(input_seq), it’s output : It is a tuple with the shape (number of examples, max number of tokens in the sequence, number of hidden units in the DistilBERT model). (2000,66.768)

![img](https://jalammar.github.io/images/distilBERT/bert-distilbert-output-tensor-predictions.png)

unpacking 3D output tensor from BERT

![img](https://jalammar.github.io/images/distilBERT/bert-output-tensor.png)

same in a sentence level

![img](https://jalammar.github.io/images/distilBERT/bert-input-to-output-tensor-recap.png)

![img](https://jalammar.github.io/images/distilBERT/bert-output-tensor-selection.png)

![img](https://jalammar.github.io/images/distilBERT/bert-output-cls-senteence-embeddings.png)