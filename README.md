# Sentiment Analysis with Social Attention

Author: Yi Yang

Contact: yangyiycc@gmail.com


## Basic Description

This is the Python implementation of the social attention model for sentiment analysis, described
in

    Yi Yang and Jacob Eisenstein "Overcoming Language Variation in Sentiment Analysis with Social Attention", TACL 2017

[[pdf]](https://arxiv.org/abs/1511.06052), [[BibTex]](#)


## Dependencies

1. [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/)
2. [Keras](https://keras.io/)
3. Optional: [CUDA Toolkit](http://docs.nvidia.com/cuda/) for GPU programming.


## Data

In order to reproduce the results reported in the paper, you will need

1. The SemEval 2015 Twitter sentiment analysis datasets, as described in [this paper](http://www.anthology.aclweb.org/S/S15/S15-2078.pdf). 
    * The data is available in the [data/txt](https://github.com/yiyang-gt/social-attention/tree/master/data/txt) folder. Unfortunately, the text content is not available due to Twitter policy. You need to replace "content" with the real tweets. 
    * You can preprocss the raw tweets using (tweet = normalizeTextForSentiment(tokenizeRawTweetText(tweet), True)), which can be found in [twokenize.py](https://github.com/yiyang-gt/social-attention/blob/master/twokenize.py).
2. The pretrained [word embeddings](https://www.l2f.inesc-id.pt/~wlin/public/embeddings/struc_skip_600.txt) (don't right click the link---use left click and Save link As...). You can save the file in [data/word_embeddings](https://github.com/yiyang-gt/social-attention/tree/master/data/word_embeddings).
3. The pretrained author embeddings, which are available in [data/author_embeddings](https://github.com/yiyang-gt/social-attention/tree/master/data/author_embeddings).


## Reproduce results

Great, now you are ready to reproduce the results

1. Prepare the data, and generate the required data file semeval.pkl (ask for the file from me via email if you need)

    python process_data.py data/word_embeddings/struc_skip_600.txt \
                           data/semeval.pkl \
                           data/txt/train_2013.txt \
                           data/txt/dev_2013.txt \
                           data/txt/test_2013.txt \
                           data/txt/test_2014.txt \
                           data/txt/test_2015.txt 

2. Reproduce *CNN* baseline results

    python cnn_baseline.py data/semeval.pkl 
 
3. Reproduce *mixture of experts* baseline results

    python mixture_expert.py data/semeval.pkl 

4. Reproduce *concatenation* baseline results

    python concat_baseline.py data/semeval.pkl data/author_embeddings/retweet.emb

5. Reproduce *SOCIAL ATTENTION* results

    python social_attention.py data/semeval.pkl data/author_embeddings/retweet.emb

