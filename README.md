This is an open source implementation of the Bilingual Distributed
Representations without Word Alignments method described in the
paper cited below.


http://arxiv.org/abs/1410.2455

BilBOWA: Fast Bilingual Distributed Representations without Word Alignments

Stephan Gouws, Yoshua Bengio, Greg Corrado


We introduce BilBOWA ("Bilingual Bag-of-Words without Alignments"),
a simple and computationally-efficient model for learning bilingual
distributed representations of words which can scale to large datasets
and does not require word-aligned training data. Instead it trains
directly on monolingual data and extracts a bilingual signal from a
smaller set of raw text sentence-aligned data. This is achieved using
a novel sampled bag-of-words cross-lingual objective, which is used to
regularize two noise-contrastive language models for efficient
cross-lingual feature learning. We show that bilingual embeddings
learned using the proposed model outperforms state-of-the-art methods
on a cross-lingual document classification task as well as a lexical
translation task on the WMT11 data. Our code will be made available as
part of an open-source toolkit.

arXiv:1410.2455
Submitted on 9 Oct 2014


## Compiling and using `Bilbowa`
To compile, do the following

    cd bilbowa
    make all

This creates `bilbowa` and `bidist` in the `bin` directory. 
Run `bilbowa` to see the list of command line options.
`bidist` is a modification of the distance module in the [word2vec package](https://github.com/danielfrg/word2vec). It receives 2 binary vectors (let's say English and French) as input, and for any given English word returns the list of the similar French words. Note that similar to the original distance module, it only works with the binary vectors (use `-binary 1` switch when calling `bilbowa` to train the system.)

