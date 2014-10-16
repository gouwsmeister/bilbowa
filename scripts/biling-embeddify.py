import cPickle
import sys
import numpy as np

if (len(sys.argv) < 7):
    print "usage: python script.py model.en model.fr en-extension fr-extension MAX_EXTRACT combined-model.pkl"
    exit(1)
# read embedding in text format into numpy array
fin1 = open(sys.argv[1], 'r')
fin2 = open(sys.argv[2], 'r')
dimX1, dimY1 = [int(e) for e in fin1.readline().split()]
dimX2, dimY2 = [int(e) for e in fin2.readline().split()]
assert dimY1 == dimY2
en_ext = '_'+sys.argv[3]
fr_ext = '_'+sys.argv[4]
MAX_EXTRACT = int(sys.argv[5])
words, embs = [], []
def extract(fin, words, embs, extension, MAX_EXTRACT):
    for i, line in enumerate(fin):
        if i >= MAX_EXTRACT:
            break
        splits = line.split()
        word, embedding = splits[0], splits[1:]
        words.append(word + extension)
        embs.append(embedding)

extract(fin1, words, embs, en_ext, MAX_EXTRACT)
extract(fin2, words, embs, fr_ext, MAX_EXTRACT)
emb_mat = np.asarray(embs, dtype='float32')
print "Extracted %d words, constructed embeddings matrix with shape %s" % (len(words), emb_mat.shape)
# save as pickle
cPickle.dump([words, emb_mat], open(sys.argv[6], 'wb'))
fin1.close()
fin2.close()

