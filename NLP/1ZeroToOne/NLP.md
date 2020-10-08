

[Full Medium article](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e)

## Step 1: Gather your data

#### data source

Every Machine Learning problem starts with data, such as a list of emails, posts, or tweets. 



#### labels

We have labeled data and so we know which tweets belong to which categories. As Richard Socher outlines , it is usually faster, simpler, and cheaper to **find and label enough data** to train a model on, rather than trying to optimize a complex unsupervised method.

## Step 2: Clean your data

> *The number one rule we follow is: “Your model will only ever be as good as your data.”*

 **A clean dataset will allow a model to learn meaningful features and not overfit on irrelevant noise**

Here is a checklist

1. Remove all **irrelevant characters** such as any non alphanumeric characters
2. **[Tokenize](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)** your text by separating it into individual words
3. Remove **words** that are **not relevant**, such as “@” twitter mentions or urls
4. Convert all **characters to lowercase**, in order to treat words such as “hello”, “Hello”, and “HELLO” the same
5. Consider **combining misspelled or alternately spelled words to a single representation** (e.g. “cool”/”kewl”/”cooool”)
6. Consider **[lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)** (reduce words such as “am”, “are”, and “is” to a common form such as “be”)

After following these steps and checking for additional errors, we can start using the clean, labelled data to train models!



## Step 3: Find a good data representation

Our dataset is a list of sentences, so in order for our algorithm to extract patterns from the data, we first need to find a way to represent it in a way that our algorithm can understand, i.e. as a list of numbers.

### One-hot encoding (Bag of Words)

A natural way to represent text for computers is to encode each character individually as a number ([ASCII](https://en.wikipedia.org/wiki/ASCII) for example). If we were to feed this simple representation into a classifier, it would have to learn the structure of words from scratch based only on our data, which is impossible for most datasets. We need to use a higher level approach.

we can build a **vocabulary** of all the unique words in our dataset, and associate a unique index to each word in the vocabulary. Each sentence is then represented as a list that is as long as the number of distinct words in our vocabulary. At each index in this list, we mark how many times the given word appears in our sentence. This is called a [**Bag of Words**](https://en.wikipedia.org/wiki/Bag-of-words_model) **model**, since it is a representation that completely ignores the order of words in our sentence.

![img](https://miro.medium.com/max/2356/1*oQ3suk0Ayc8z8i1QIl5Big.png)



### Visualizing the embeddings

We have around 20,000 words in our vocabulary in the “Disasters of Social Media” example, which means that every sentence will be represented as a vector of length 20,000. The vector will contain **mostly 0s** because each sentence contains only a very small subset of our vocabulary.

n order to see whether our embeddings are capturing information that is **relevant to our problem**. it is a good idea to visualize them and see if the classes look well separated. Since vocabularies are usually very large and visualizing data in 20,000 dimensions is impossible, techniques like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) will help project the data down to two dimensions.

In order to see whether the Bag of Words features are of any use, we can train a classifier based on them.



## Step 4: Classification

When first approaching a problem, a general best practice is to start with the simplest tool that could solve the job. Whenever it comes to classifying data, a common favorite for its versatility and explainability is [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression). It is very simple to train and the results are interpretable as you can easily extract the most important coefficients from the model.

We split our data in to a training set used to fit our model and a test set to see how well it generalizes to unseen data. After training, we get an **accuracy of 75.4%.** Not too shabby! Guessing the most frequent class (“irrelevant”) would give us only 57%. However, even if 75% precision was good enough for our needs, **we should never ship a model without trying to understand it.**



## Step 5: Inspection

### Confusion Matrix

A first step is to understand the types of errors our model makes, and which kind of errors are least desirable. In our example, **false positives** are classifying an irrelevant tweet as a disaster, and **false negatives** are classifying a disaster as an irrelevant tweet. If the priority is to react to every potential event, we would want to lower our false negatives. If we are constrained in resources however, we might prioritize a lower false positive rate to reduce false alarms. A good way to visualize this information is using a [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix), which compares the predictions our model makes with the true label. Ideally, the matrix would be a diagonal line from top left to bottom right (our predictions match the truth perfectly).

<img src="https://miro.medium.com/max/1388/1*DicbwUOoqFezDWROfsZ-CQ.png" alt="img" style="zoom:33%;" />

Our classifier creates more false negatives than false positives (proportionally). In other words, our model’s most common error is inaccurately classifying disasters as irrelevant. If false positives represent a high cost for law enforcement, this could be a good bias for our classifier to have.



### Explaining and interpreting our model

To validate our model and interpret its predictions, it is important to look at which words it is using to make decisions. If our data is biased, our classifier will make accurate predictions in the sample data, but the model would not generalize well in the real world. Here we plot the **most important words** for both the disaster and irrelevant class. Plotting word importance is simple with Bag of Words and Logistic Regression, since we can just extract and rank the coefficients that the model used for its predictions.

Our classifier correctly picks up on some patterns (hiroshima, massacre), but clearly seems to be overfitting on some meaningless terms (heyoo, x1392). Right now, our Bag of Words model is dealing with a huge vocabulary of different words and **treating all words equally**. However, some of these words are very frequent, and are only contributing noise to our predictions. Next, we will try a way to represent sentences that can account for the frequency of words, to see if we can pick up more signal from our data.



## Step 6: Accounting for vocabulary structure

### TF-IDF

In order to help our model focus more on meaningful words, we can use a [TF-IDF score](https://en.wikipedia.org/wiki/Tf–idf) (Term Frequency, Inverse Document Frequency) on top of our Bag of Words model. TF-IDF weighs words by how rare they are in our dataset, discounting words that are too frequent and just add to the noise. Here is the PCA projection of our new embeddings.

The words it picked up look much more relevant! Although our metrics on our test set only increased slightly, we have much more confidence in the terms our model is using, and thus would feel more comfortable deploying it in a system that would interact with customers.



## Step 7: Leveraging semantics

### Word2Vec

Our latest model managed to pick up on high signal words. However, it is very likely that if we deploy this model, we will encounter words that we have not seen in our training set before. The previous model will not be able to accurately classify these tweets, **even if it has seen very similar words during training**.

To solve this problem, we need to capture the **semantic meaning of words**, meaning we need to understand that words like ‘good’ and ‘positive’ are closer than ‘apricot’ and ‘continent.’ The tool we will use to help us capture meaning is called **Word2Vec.**

#### **Using pre-trained words**

[Word2Vec](https://arxiv.org/abs/1301.3781) is a technique to find continuous embeddings for words. It learns from reading massive amounts of text and memorizing which words tend to appear in similar contexts. After being trained on enough data, it generates a 300-dimension vector for each word in a vocabulary, with words of similar meaning being closer to each other.

The authors of the [paper](https://arxiv.org/abs/1301.3781) open sourced a model that was pre-trained on a very large corpus which we can leverage to include some knowledge of semantic meaning into our model. The pre-trained vectors can be found in the [repository](https://github.com/hundredblocks/concrete_NLP_tutorial) associated with this post.

### Sentence level representation

A quick way to get a sentence embedding for our classifier is to average Word2Vec scores of all words in our sentence. This is a Bag of Words approach just like before, but this time **we only lose the syntax of our sentence, while keeping some semantic information.**

![img](https://miro.medium.com/max/2306/1*THo9NKchWkCAOILvs1eHuQ.png)

our new embeddings should help our classifier find the separation between both classes. After training the same model a third time (a Logistic Regression), we get **an accuracy score of 77.7%**, our best result yet! Time to inspect our model.

#### Complexity/Explainability trade-off

Since our embeddings are not represented as a vector with one dimension per word as in our previous models, it’s harder to see which words are the most relevant to our classification. While we still have access to the coefficients of our Logistic Regression, they relate to the 300 dimensions of our embeddings rather than the indices of words.

For such a low gain in accuracy, losing all explainability seems like a harsh trade-off. However, with more complex models we can leverage **black box explainers** such as [LIME](https://arxiv.org/abs/1602.04938) in order to get some insight into how our classifier works.

##### LIME

LIME is [available on Github](https://github.com/marcotcr/lime) through an open-sourced package. A black-box explainer allows users to explain the decisions of any classifier **on one particular example** by perturbing the input (in our case removing words from the sentence) and seeing how the prediction changes.

## Step 8: Leveraging syntax using end-to-end approaches

We’ve covered quick and efficient approaches to generate compact sentence embeddings. However, by omitting the order of words, we are discarding all of the syntactic information of our sentences. If these methods do not provide sufficient results, you can utilize more complex model that take in whole sentences as input and predict labels without the need to build an intermediate representation. A common way to do that is to treat a sentence as **a sequence of individual word vectors** using either Word2Vec or more recent approaches such as **[GloVe](https://nlp.stanford.edu/projects/glove/)** or [CoVe](https://arxiv.org/abs/1708.00107).

![img](https://miro.medium.com/max/2982/1*P6YbB_gmgXmosjfQ-QB5XA.png)

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) train very quickly and work well as an entry level deep learning architecture. While Convolutional Neural Networks (CNN) are mainly known for their performance on image data, they have been providing excellent results on text related tasks, and are usually much quicker to train than most complex NLP approaches (e.g. [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [Encoder/Decoder](https://www.tensorflow.org/tutorials/seq2seq) architectures). This model preserves the order of words and learns valuable information on which sequences of words are predictive of our target classes. Contrary to previous models, it can tell the difference between “Alex eats plants” and “Plants eat Alex.”

