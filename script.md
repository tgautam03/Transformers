# Motivation
The year was 2017 when a research paper introducing Transformers was published. Looking back, this was the moment that changed the field of AI forever as it led to the development of Generative Pre-trained Transformers or GPT for short. Yes, ChatGPT is an application that has GPT-3 at it’s core. All this might look intimidating, but the concepts behind Transformer are very simple.

This video has two main goals:
- Explain Transformers and it’s various components, and then
- Build a simple Transformers based classifier from scratch

So without further ado let’s get started...

# Self Attention
## Dataset
At it’s core, Transformers get their super powers from a mechanism called Self Attention. In simple terms, Self Attention is a sequence to sequence operation. What I mean by this is that, it takes in a sequence (which can be a sentence or a series of number) and returns a different sequence. How exactly that happens, let’s look at the details. 

Everything starts with the dataset. For the illustration, I’m using the simple IMDB dataset for classifying a review as positive or negative. There are two parts to this dataset, the actual review in raw text form and a corresponding label mentioning whether its a good or a bad review. The number one in this case means that the review is actually good. This is just one example, for this problem we have a total of 20k such reviews, some good and some bad.

We know that the computers can't make sense of text. The only thing they understand is numbers. So we have to find a way to represent each word as a number. A very common way to do that is to collect all the individual words in the dataset and store them to create a vocabulary. Now we can simply assign all of them a unique number. Looking at this vocabulary, we can map any review to a set of numbers. So, now we have a bunch of sequences of numbers that esentially represent a sentence. But there's a problem with this. All these are of different lengths and to feed these into a transformer and utilize the parallelism, these have to be of same lengths. This is done by simply padding each sequence with zeros at the end.

```py
vocabulary1 = ' '.join(word for word in vocab.get_itos()[2:95])
vocabulary2 = ' '.join(word for word in vocab.get_itos()[95:202])

import pickle

with open('../anim/vocab1.pkl', 'wb') as fp:
    pickle.dump(vocabulary1, fp)
with open('../anim/vocab2.pkl', 'wb') as fp:
    pickle.dump(vocabulary2, fp)

num_vocabulary1 = list(map(str, np.arange(0, 121).tolist()))
num_vocabulary2 = list(map(str, np.arange(len(vocab)-114, len(vocab)).tolist()))

num_vocabulary1 = ' '.join(word for word in num_vocabulary1)
num_vocabulary2 = ' '.join(word for word in num_vocabulary2)

import pickle

with open('../anim/num_vocab1.pkl', 'wb') as fp:
    pickle.dump(num_vocabulary1, fp)
with open('../anim/num_vocab2.pkl', 'wb') as fp:
    pickle.dump(num_vocabulary2, fp)
```

To undertand how self attention works, let's just look at one of these sequences and only consider the first five words. Now representing each word with a single number is ok, but it doesn't convey much information. It's a lot better if we can write each word as a vector of say length 7 rather than just a single number. I'm picking 7 as an arbitrary number and you can choose something else we well. This can be done using Embedding class from PyTorch itself. Now, there are a lot of things that I can say about this Embedding class, but I won't go down the rabbit hole and for the purpose of this video, just think of this as another unique mapping from an integer to a floating point vector of desired length based on the vocabulary size. So, just to recap, we have now transformed and stored the review in form of a tensor of shape 1 x 5 x 7, where 1 is because I'm only looking at 1 review and 5 is for the number of words in this review and 7 is the embedding dimension. By the way, if you want to check out the code 

## Self Attention Operation
This is where Self Attention Operation comes in and works on the dataset preprocessed and represented in this form. If you remember, self attention is just a sequence to sequence operation. In this case these sequences are individual words. What self attention does is takes the weighted average of each of these words. This can be seen as premultiplying the input matrix by a weight matrix. The output matrix is then of the same size as that of the input matrix. Now we have a way to get the output but what about this weight matrix. The weight matrix is calculated by simply multiplying the input matrix with it's transpose. In this case input matrix is 7 elements wide but it can be longer, so it's better to scale the output by dividing it with the square root of the embedding dimension. Finally, as it is a weightage matrix, we would like each row to sum to 1. This can be done by applying softmax to each row of this matrix.

Writing this as a python function is pretty straight forward. For the given input tensor X of shape 1 x 5 x 7, we can use pytorch to define matrix multiplications and return the output. Now there is a problem here. If I change the sequence of words in the sentence, it can alter the meaning of a sentence. However, the output from self attention does not change. It's only rearranged in the same order as the inputs. This is problamatic as the self attention operation does not seem to care too much about the order of the words in a sentence! This can be solved by encorporating positional encoding. Positional encoding is done by simply adding another matrix which contains the information related to the position of the words in a sentence and adding it to the original input matrix. This position matrix can be generated using pytorch's embedding layer as well! All we have to do is define the embedding dimension, which is 7 in this case. Max number of words in a sentence that you anticipate and then we can just generate the matrix and reshape it to match the shape of our input martrix.

## Bringing Learning to Self Attention
As you can see that the Self Attention operation isn't that complicated. However, you might be thinking, that where is Machine Learning??? Well, in theory Self Attention has nothing to do with Machine Learning but we can modify this framework to encorporate Machine Learning. Let's see how we can do that...

If you look carefully, there are three ways in which the input matrix X is being used in self attention. Instead of using X as it is, if we send this matrix through three separate neural nets and then apply self attention to the outputs, we can incorporate learning from data in the self attention. By the way, these three output matrices are called Query, Key and Value! Transformers have some historical baggage associated with them and that lead to this Query, Key and Value name. In any case, it's irrelevant as far as the mathematics or the technique is concerned. So finally with this we have a self attention mechanism that can actually learn from data.