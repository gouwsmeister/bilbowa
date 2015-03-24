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
enwords, enembs = [], []
frwords, frembs = [], []
def extract(fin, words, embs, extension, MAX_EXTRACT):
    for i, line in enumerate(fin):
        if MAX_EXTRACT > 0 and i >= MAX_EXTRACT:
            break
        splits = line.split()
        word, embedding = splits[0], splits[1:]
        words.append(word + extension)
        embs.append(embedding)
    if (MAX_EXTRACT < 0):
        # take first and last MAX_EXTRACT embs
        print "Clipping to first and last ", abs(MAX_EXTRACT)
        #embs = embs[0:abs(MAX_EXTRACT)] + embs[MAX_EXTRACT:]
        #words = words[0:abs(MAX_EXTRACT)] + words[MAX_EXTRACT:]
        del embs[abs(MAX_EXTRACT):MAX_EXTRACT]  #inplace
        del words[abs(MAX_EXTRACT):MAX_EXTRACT]

extract(fin1, enwords, enembs, en_ext, MAX_EXTRACT)
extract(fin2, frwords, frembs, fr_ext, MAX_EXTRACT)
embs = enembs + frembs
words = enwords + frwords
emb_mat = np.asarray(embs, dtype='float32')
print "Extracted %d words, constructed embeddings matrix with shape %s" % (len(words), emb_mat.shape)
print "Average embedding norms: %f" % (np.sqrt((emb_mat**2).sum(axis=1)).mean(),)
# save as pickle
cPickle.dump([words, emb_mat], open(sys.argv[6], 'wb'))
fin1.close()
fin2.close()

