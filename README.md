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
