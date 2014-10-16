//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SEN_LEN 1000
#define MAX_CODE_LENGTH 40

#define NUM_LANG 2
#define CLIP_UPDATES 0.1                // biggest update per parameter per step
#define PAIRS_TO_SAMPLE 100             // number of cross-lingual word-pairs to sample per sentence

const int vocab_hash_size = 30000000;   // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;                     // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char *mono_train_files[NUM_LANG],
     *par_train_files[NUM_LANG], 
     *output_files[NUM_LANG], 
     *save_vocab_files[NUM_LANG], 
     *read_vocab_files[NUM_LANG];
struct vocab_word *vocabs[NUM_LANG];
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, 
    num_threads = 1, min_reduce = 1;
int *vocab_hashes[NUM_LANG];
long long vocab_max_size = 1000, vocab_sizes[NUM_LANG], layer1_size = 40;
long long train_words[NUM_LANG], word_count_actual, file_sizes[NUM_LANG];
long long lang_updates[NUM_LANG], dump_every=0, dump_iters[NUM_LANG],
     epoch[NUM_LANG];
unsigned long long next_random = 0;
int learn_vocab_and_quit = 0, PLUSPLUS = 0, MONO_SAMPLE = 1, PAR_SAMPLE = 1, adagrad = 0;
real alpha = 0.025, starting_alpha, sample = 0, ALIGN_LAMBDA = 1.0, bilbowa_grad = 0;
real *syn0s[NUM_LANG], *syn1s[NUM_LANG], *syn1negs[NUM_LANG], 
	 *syn0grads[NUM_LANG], *syn1negGrads[NUM_LANG], *expTable;
clock_t start;

const int table_size = 1e8;     // const across languages
int *tables[NUM_LANG];
int negative = 15, MONO_DONE_TRAINING = 0;
long long NUM_EPOCHS = 1, EARLY_STOP = 0, max_train_words;
real *delta_pos, XLING_LAMBDA = 1.0;

void InitUnigramTable(int lang_id) {
  int a, i;
  long long train_words_pow = 0, vocab_size = vocab_sizes[lang_id];
  int *table;
  struct vocab_word *vocab = vocabs[lang_id];
  real d1, power = 0.75;
  table = tables[lang_id] = malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

/* Reads a single word from a file, assuming space + tab + EOL to be word 
   boundaries */
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

/* Returns hash value of a word */
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

/* Returns position of a word in the vocabulary; if the word is not found, 
 * returns -1 */
int SearchVocab(int lang_id, char *word) {
  unsigned int hash = GetWordHash(word);
  //if (lang_id >= NUM_LANG) { printf("lang_id >= NUM_LANG\n"); exit(1); }
  int *vocab_hash = vocab_hashes[lang_id];
  struct vocab_word *vocab = vocabs[lang_id];
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

/* Reads a word and returns its index in the vocabulary */
int ReadWordIndex(FILE *fin, int lang_id) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(lang_id, word);         // MOD
}


/* Adds a word to the vocabulary */
int AddWordToVocab(int lang_id, char *word) {    
  unsigned int hash, length = strlen(word) + 1;
  struct vocab_word *vocab = vocabs[lang_id];
  int *vocab_hash = vocab_hashes[lang_id];			// array of *ints
  
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_sizes[lang_id]].word = calloc(length, sizeof(char));
  strcpy(vocab[vocab_sizes[lang_id]].word, word);
  vocab[vocab_sizes[lang_id]].cn = 0;
  vocab_sizes[lang_id]++;
  // Reallocate memory if needed
  if (vocab_sizes[lang_id] + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocabs[lang_id] = (struct vocab_word *)realloc(vocabs[lang_id], 
        vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_sizes[lang_id] - 1;
  return vocab_sizes[lang_id] - 1;
}

/* Used later for sorting by word counts */
int VocabCompare(const void *word1, const void *word2) {
    return ((struct vocab_word *)word2)->cn - ((struct vocab_word *)word1)->cn;
}

/* Sorts the vocabulary by frequency using word counts */
void SortVocab(int lang_id) {
  int a, size;
  unsigned int hash;
  struct vocab_word *vocab = vocabs[lang_id];
  int *vocab_hash = vocab_hashes[lang_id];

  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_sizes[lang_id] - 1, sizeof(struct vocab_word),
      VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_sizes[lang_id];
  train_words[lang_id] = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_sizes[lang_id]--;
      free(vocab[vocab_sizes[lang_id]].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not correct
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words[lang_id] += vocab[a].cn;
    }
  }
  vocabs[lang_id] = (struct vocab_word *)realloc(vocabs[lang_id], 
      (vocab_sizes[lang_id] + 1) * sizeof(struct vocab_word));
}

/* Reduces the vocabulary by removing infrequent tokens */
void ReduceVocab(int lang_id) {
  int a, b = 0;
  unsigned int hash;
  long long vocab_size = vocab_sizes[lang_id];
  struct vocab_word *vocab = vocabs[lang_id];
  int *vocab_hash = vocab_hashes[lang_id];

  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_sizes[lang_id] = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not correct
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile(int lang_id) {
  char word[MAX_STRING], *train_file = mono_train_files[lang_id];
  FILE *fin;
  long long a, i;
  int *vocab_hash = vocab_hashes[lang_id];
  struct vocab_word *vocab = vocabs[lang_id];
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data (%s) file not found (lang_id==%d)!\n", 
        train_file, lang_id);
    exit(1);
  }
  vocab_sizes[lang_id] = 0;
  AddWordToVocab(lang_id, (char *)"</s>");
  vocab = vocabs[lang_id];
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words[lang_id]++;
    // learn only on the first EARLY_STOP words if the flag is set
    if (EARLY_STOP > 0 && train_words[lang_id] > EARLY_STOP) break; 
    if ((debug_mode > 1) && (train_words[lang_id] % 100000 == 0)) {
      printf("%lldK%c", train_words[lang_id] / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(lang_id, word);
    if (i == -1) {
      a = AddWordToVocab(lang_id, word);
      vocab = vocabs[lang_id];      // might have changed
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_sizes[lang_id] > vocab_hash_size * 0.7) {
	  ReduceVocab(lang_id);     
	}
  }
  fprintf(stderr, "pre SortVocab\n");
  SortVocab(lang_id);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_sizes[lang_id]);
    printf("Words in train file: %lld\n", train_words[lang_id]);
  }
  file_sizes[lang_id] = ftell(fin);
  fclose(fin);
}

void SaveVocab(int lang_id) {
  long long i;
  char *save_vocab_file = save_vocab_files[lang_id];
  struct vocab_word *vocab = vocabs[lang_id];

  FILE *fo = fopen(save_vocab_file, "wb");
  printf("Saving vocabulary with %lld entries to %s\n", vocab_sizes[lang_id], 
      save_vocab_file);
  for (i = 0; i < vocab_sizes[lang_id]; i++) fprintf(fo, "%s %lld\n", 
      vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab(int lang_id) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  int *vocab_hash = vocab_hashes[lang_id];
  char *train_file = mono_train_files[lang_id];
  FILE *fin = fopen(read_vocab_files[lang_id], "rb");

  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_sizes[lang_id] = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(lang_id, word);      // can change vocabs
    fscanf(fin, "%lld%c", &vocabs[lang_id][a].cn, &c);
    i++;
  }
  SortVocab(lang_id);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_sizes[lang_id]);
    printf("Words in train file: %lld\n", train_words[lang_id]);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file (%s) not found!\n", train_file);
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_sizes[lang_id] = ftell(fin);
  fclose(fin);
}

void InitNet(int lang_id) {
  long long a, b, vocab_size = vocab_sizes[lang_id];
  real *syn0, *syn1neg, *syn0grad, *syn1negGrad;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size *
      sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn0s[lang_id] = syn0;
  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * 
      layer1_size * sizeof(real));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn1negs[lang_id] = syn1neg;
  if (adagrad) {
	  a = posix_memalign((void **)&syn0grad, 128, (long long)vocab_size * 
        layer1_size * sizeof(real));
  	if (syn0grad == NULL) {printf("Memory allocation failed\n"); exit(1);}
  	else syn0grads[lang_id] = syn0grad;
  	a = posix_memalign((void **)&syn1negGrad, 128, (long long)vocab_size * 
        layer1_size * sizeof(real));
  	if (syn1negGrad == NULL) {printf("Memory allocation failed\n"); exit(1);}
	else syn1negGrads[lang_id] = syn1negGrad;
  }
  for (b = 0; b < layer1_size; b++) { 
    for (a = 0; a < vocab_size; a++) {
      syn1neg[a * layer1_size + b] = 0;
      syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	  if (adagrad) {
		syn0grad[a*layer1_size + b] = 0;
	    syn1negGrad[a*layer1_size + b] = 0;
	  }
    }
  } 
}

char SubSample(int lang_id, long long word_id) {
  long long count = vocabs[lang_id][word_id].cn;
  real thresh = (sqrt(count / (sample * train_words[lang_id])) + 1) * 
    (sample * train_words[lang_id]) / count;
  next_random = next_random * (unsigned long long)25214903917 + 11;
  if ((next_random & 0xFFFF) / (real)65536 > thresh) return 1;
  else return 0;
}

/* Read a sentence into *sen using vocabulary for language lang_id
 * Store processed words in *sen, returns (potentially subsampled) 
 * length of sentence */
int ReadSent(FILE *fi, int lang_id, long long *sen, char subsample) {
  long long word;
  int sentence_length = 0;
  //struct vocab_word *vocab = vocabs[lang_id];
  while (1) {
    word = ReadWordIndex(fi, lang_id);
    if (feof(fi)) break;
    if (word == -1) continue;       // unknown
    if (word == 0) break;           // end-of-sentence
    // The subsampling randomly discards frequent words while keeping the 
    // ranking the same.
    if (subsample && sample > 0) {
      if (SubSample(lang_id, word)) continue;
    }
    sen[sentence_length] = word;
    sentence_length++;
    if (sentence_length >= MAX_SEN_LEN) break;
  }
  return sentence_length;
}

void UpdateEmbeddings(real *embeddings, real *grads, int offset, 
					  int num_updates, real *deltas, real weight) {
  int a;
  real step, epsilon = 1e-6;
  for (a = 0; a < num_updates; a++) {
    if (adagrad) {
      // Use Adagrad for automatic learning rate selection
      grads[offset + a] += (deltas[a] * deltas[a]);
      step = (alpha / fmax(epsilon, sqrt(grads[offset + a]))) * deltas[a];
    } else {
      // Regular SGD
      step = alpha * deltas[a];
    }
    if (step != step) {
      fprintf(stderr, "ERROR: step == NaN\n");
    }
      step = step * weight;
      if (CLIP_UPDATES != 0) {
        if (step > CLIP_UPDATES) step = CLIP_UPDATES;
        if (step < -CLIP_UPDATES) step = -CLIP_UPDATES;
      }
      embeddings[offset + a] += step;   
    }
}

void UpdateEnFrSquaredError(int en_sen_len, int fr_sen_len, 
    long long *en_sen, long long *fr_sen, real *delta, real *syn0_e, 
    real *syn0_f, real weight) {
  int d, offset;
  // To minimize squared error:
  // delta = d .5*|| e - f ||^2 = ±(e - f)
  // d/den = +delta 
  for (d = 0; d < en_sen_len; d++) {
    offset = layer1_size * en_sen[d];
	  // update in -d/den = -delta direction
	  UpdateEmbeddings(syn0_e, syn0grads[0], offset, layer1_size, delta, -weight);
  }
  // d/df = -delta
  for (d = 0; d < fr_sen_len; d++) {
    offset = layer1_size * fr_sen[d];
	  // update in -d/df = +delta direction
	  UpdateEmbeddings(syn0_f, syn0grads[1], offset, layer1_size, delta, weight);
  }
}

real FpropSent(int len, long long *sen, real *deltas, real *syn, real sign) {
  real sumSquares = 0;
  long long c, d, offset;
  for (d = 0; d < len; d++) {
     offset = layer1_size * sen[d];
     for (c = 0; c < layer1_size; c++) {
       // We compute the MEAN sentence vector
       deltas[c] += sign * syn[offset + c] / (real)len;    
       sumSquares += deltas[c] * deltas[c];
     }
  }
  return sumSquares;
}

void BuildMonoCDF(real *cdf, long long *sen, int lang_id, int len) {
  int i;
  long long word, count;
  real threshold, normalizer = 0;
  for (i = 0; i < len; i++) {
    word = sen[i];
    if (word <= 0) {                   // unk / <s>
      threshold = 0.0;
    }
    else if (PAR_SAMPLE && sample > 0) {  // subsample
      count = vocabs[lang_id][word].cn;
      threshold = (sqrt(count / (sample * train_words[lang_id])) + 1) * (sample *
        train_words[lang_id]) / count; 
      if (threshold < 0) threshold = 0.0;
    } 
    else threshold = 1.0;                    // else uniform
    normalizer += threshold;
    cdf[i] = threshold;
    if (i > 0) cdf[i] += cdf[i - 1];
  }
  //printf("------------\n");
  for (i = 0; i < len; i++) {
    cdf[i] /= normalizer;
    //printf("%.4f | %s\n", cdf[i], vocabs[lang_id][*(&sen[i])].word);
  }
  //if (cdf[N - 1] != 1.0) 
  //  printf("Last element in CDF != 1.0 (%.4f)\n", cdf[N - 1]);

}


/* Construct a cumulative distribution function (CDF) in *cdf for the current
 * parallel sentence pair for drawing a sample from the (m x n) word-alignment
 * matrix. For the first language, set len2 == -1. This constructs an m-length,
 * uniform sampling CDF. 
 * For the second language, specify -ALIGN_LAMBDA > 0 during runtime to 
 * upweight words closer to pos1 on the diagonal, else a uniform cdf is 
 * returned.
 * For both cases, to enable "subsampling", set PAR_SAMPLE == 1 and
 * specify -sample > 0 during runtime. With subsampling, the probability 
 * for a word to be sampled will be downweighted proportional to 
 * its unigram frequency of occurrence.
 */
void BuildBilingCDF(real *cdf, long long *sen, int lang_id, int pos1, int len1, 
    int len2) {
  int i, N;
  long long word, count;
  real threshold, normalizer = 0;
  if (len2 == -1) N = len1; else N = len2;
  for (i = 0; i < N; i++) {
    word = sen[i];
    if (word <= 0) {             // unk / <s>
      threshold = 0.0;           // don't sample UNKs
    }
    else if (PAR_SAMPLE && sample > 0) {  // subsample
      count = vocabs[lang_id][word].cn;
      threshold = (sqrt(count / (sample * train_words[lang_id])) + 1) * (sample *
        train_words[lang_id]) / count; 
      if (threshold < 0) threshold = 0.0;
    } 
    else threshold = 1.0;                    // else uniform
    if (len2 > 0) {
      // set ALIGN_LAMBDA=0 for uniform alignment, else diagonal model
      threshold *= exp(-ALIGN_LAMBDA * fabs(pos1/(real)len1 - i/(real)len2));
    }
    normalizer += threshold;
    cdf[i] = threshold;
    if (i > 0) cdf[i] += cdf[i - 1];
  }
  //printf("------------\n");
  for (i = 0; i < N; i++) {
    cdf[i] /= normalizer;
    //printf("%.4f | %s\n", cdf[i], vocabs[lang_id][*(&sen[i])].word);
  }
  //if (cdf[N - 1] != 1.0) 
  //  printf("Last element in CDF != 1.0 (%.4f)\n", cdf[N - 1]);
}

/* BilBOWA++ Monte Carlo update */
void BilBOWAPlusPlusMCUpdate(int par_sen_len1, int par_sen_len2,
    long long *par_sen1, long long *par_sen2, real xling_weight,
    real *deltas, real *syn0_e, real *syn0_f) { 
  int i, j, pairs_updated = 0;
  real ensample[MAX_SEN_LEN], frsample[MAX_SEN_LEN], thresh, grad_norm; 
  BuildBilingCDF(ensample, par_sen1, 0, -1, par_sen_len1, -1);
  if (ALIGN_LAMBDA == 0)
    BuildBilingCDF(frsample, par_sen2, 1, -1, par_sen_len1, par_sen_len2); // doesn't change
  while (pairs_updated < PAIRS_TO_SAMPLE) {
    // SAMPLE en WORD
    thresh = rand() / (real)RAND_MAX;
    for (i = 0; i < par_sen_len1 && ensample[i] < thresh; i++);
    // SAMPLE fr WORD
    if (ALIGN_LAMBDA > 0) 
      BuildBilingCDF(frsample, par_sen2, 1, i, par_sen_len1, par_sen_len2);
    thresh = rand() / (real)RAND_MAX;
    for (j = 0; j < par_sen_len2 && frsample[j] < thresh; j++);
    // PERFORM CROSS-LINGUAL UPDATE
    memset(deltas, 0, sizeof(real) * layer1_size);
    FpropSent(1, &par_sen1[i], deltas, syn0_e, +1);
    grad_norm = FpropSent(1, &par_sen2[j], deltas, syn0_f, -1);
    bilbowa_grad = 0.9 * bilbowa_grad + 0.1 * grad_norm;
    //printf("%s (en) <--> ", vocabs[0][*(&par_sen1[i])].word);
    //printf("%s (fr)", vocabs[1][*(&par_sen2[j])].word);
    //printf(" ||grad|| = %.4f \n", grad_norm);
    UpdateEnFrSquaredError(1, 1, &par_sen1[i], &par_sen2[j], deltas, 
        syn0_e, syn0_f, XLING_LAMBDA * xling_weight);
    pairs_updated++;
  }
  return;
}


/* BilBOWA bag-of-words sentence update */
void BilBOWASentenceUpdate(int par_sen_len1, int par_sen_len2,
    long long *par_sen1, long long *par_sen2, real xling_balancer,
    real *deltas, real *syn0_e, real *syn0_f) {
  int a;
  real grad_norm;
  // FPROP
  // RESET DELTAS
  for (a = 0; a < layer1_size; a++) deltas[a] = 0;
  // ACCUMULATE L2 LOSS DELTA
  FpropSent(par_sen_len1, par_sen1, deltas, syn0_e, +1); 
  grad_norm = FpropSent(par_sen_len2, par_sen2, deltas, syn0_f, -1);
  bilbowa_grad = 0.9*bilbowa_grad + 0.1*grad_norm;
  //printf("||grad|| = %.4f\n xling_balancer = %.4f\n", XLING_LAMBDA *
  //    xling_balancer * grad_norm, xling_balancer);
  // BPROP
  UpdateEnFrSquaredError(par_sen_len1, par_sen_len2, 
    par_sen1, par_sen2, deltas, syn0_e, syn0_f, 
    XLING_LAMBDA * xling_balancer);
}

/* Thread for performing the cross-lingual learning */
void *BilbowaThread(void *id) {
  int par_sen_len1, par_sen_len2, 
      lang_id1 = (int)id / num_threads, 
      lang_id2 = (int)id / num_threads + 1,
      thread_id = (int)id % num_threads; // total_sampled;
  long long par_sen1[MAX_SEN_LEN], par_sen2[MAX_SEN_LEN],
    //        sampled_sen1[10], sampled_sen2[10],
            updates_l1 = 1, updates_l2 = 1,
            f1_size, f2_size;
  real *syn0_e = syn0s[lang_id1], *syn0_f = syn0s[lang_id2];
  real deltas[layer1_size], xling_balancer;
  //real threshold, ensample[MAX_SEN_LEN], frsample[MAX_SEN_LEN];
  FILE *fi_par1, *fi_par2;  

  //printf("BilBOWA thread: *id==%d, lang_id1==%d, lang_id2==%d, thread_id==%d\n", 
  //    (int)id, lang_id1, lang_id2, thread_id);
  fi_par1 = fopen(par_train_files[lang_id1], "rb");   // en
  fseek(fi_par1, 0, SEEK_END); 
  f1_size = ftell(fi_par1); 
  fseek(fi_par1, f1_size / num_threads * thread_id, SEEK_SET);
  fseek(fi_par1, 0, SEEK_SET);
  fi_par2 = fopen(par_train_files[lang_id2], "rb");   // fr 
  fseek(fi_par2, 0, SEEK_END); 
  f2_size = ftell(fi_par2); 
  fseek(fi_par2, f2_size / num_threads * thread_id, SEEK_SET);

  // Continue training while monolingual models are still training
  while (MONO_DONE_TRAINING < NUM_LANG * num_threads) {
    par_sen_len1 = ReadSent(fi_par1, lang_id1, par_sen1, 1);
    par_sen_len2 = ReadSent(fi_par2, lang_id2, par_sen2, 1);
    /*for (i = 0; i < par_sen_len1; i++)
      printf("%s ", vocabs[0][*(&par_sen1[i])].word);
    printf("\n");
    for (i = 0; i < par_sen_len2; i++) 
      printf("%s ", vocabs[1][*(&par_sen2[i])].word);
    printf("\n"); */
    if (feof(fi_par1) || feof(fi_par2) ||
        ftell(fi_par1) > f1_size / num_threads * (thread_id + 1) ||
        ftell(fi_par2) > f2_size / num_threads * (thread_id + 1)) {
      // recycle parallel sentences
      fseek(fi_par1, f1_size / num_threads * thread_id, SEEK_SET);
      fseek(fi_par2, f2_size / num_threads * thread_id, SEEK_SET);
      continue;
    }
    //xling_balancer = (lang_updates[0] + lang_updates[1]) / 
    //  (real)(updates_l1 + updates_l2);
    xling_balancer = 1.0;
    if (!PLUSPLUS) {    // sentence-vector
      // TODO: Move this out to BilBOWASentenceUpdate() or
      // remove altogether
      /*
      BuildCDF(ensample, par_sen1, 0, par_sen_len1);
      BuildCDF(frsample, par_sen2, 1, par_sen_len2);
      total_sampled = 0;
      //printf("--------\n");
      while (total_sampled < window * 2) {
        threshold = rand() / (real)RAND_MAX;
        for (i = 0; i < par_sen_len1 && ensample[i] < threshold; i++);
        for (j = 0; j < par_sen_len2 && frsample[j] < threshold; j++);
        sampled_sen1[total_sampled] = par_sen1[i];
        sampled_sen2[total_sampled] = par_sen2[j];
        //printf("%s (en) <--> ", vocabs[0][*(&par_sen1[i])].word);
        //printf("%s (fr)\n", vocabs[1][*(&par_sen2[j])].word);
        //printf(" ||grad|| = %.4f \n", grad_norm);
        total_sampled++;
      }
      BilBOWASentenceUpdate(total_sampled, total_sampled, sampled_sen1, sampled_sen2, 
        xling_balancer, deltas, syn0_e, syn0_f);
      */
      BilBOWASentenceUpdate(par_sen_len1, par_sen_len2, par_sen1, par_sen2, 
        xling_balancer, deltas, syn0_e, syn0_f);
    } else {
      // word vector
      BilBOWAPlusPlusMCUpdate(par_sen_len1, par_sen_len2, par_sen1, par_sen2, 
          xling_balancer, deltas, syn0_e, syn0_f);
    }
    updates_l1 += par_sen_len1; //total_sampled;
    updates_l2 += par_sen_len2; //total_sampled;
  } // while training loop
  fclose(fi_par1);
  fclose(fi_par2);
  pthread_exit(NULL);
}

void SaveModel(int lang_id, char *name) {
  long a, b;
  struct vocab_word *vocab = vocabs[lang_id];
  real *syn0 = syn0s[lang_id];
  FILE *fo = fopen(name, "wb");

  printf("\nSaving model to file: %s\n", name);
  fprintf(fo, "%lld %lld\n", vocab_sizes[lang_id], layer1_size);
  for (a = 0; a < vocab_sizes[lang_id]; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) 
      fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else 
      for (b = 0; b < layer1_size; b++) 
        fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

/* Monolingual training thread */ 
void *MonoModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, 
       sentence_position = 0;
  long long word_count = 0, last_word_count = 0, all_train_words = 0;
  long long mono_sen[MAX_SEN_LEN + 1];
  long long l1, l2, c, target, label;
  int lang_id = (int)id / num_threads, thread_id = (int)id % num_threads;
  char *train_file = mono_train_files[lang_id];
  long long vocab_size = vocab_sizes[lang_id];
  real f, g;
  clock_t now;
  real *neu1 = calloc(layer1_size, sizeof(real));
  real *neu1e = calloc(layer1_size, sizeof(real));
  real *syn1neg = syn1negs[lang_id];
  real *syn1negDelta = calloc(layer1_size, sizeof(real));
  real *syn0 = syn0s[lang_id];
  FILE *fi = fopen(train_file, "rb");

  if (!EARLY_STOP)
    // If two languages have different amounts of training data,
    // recycle the smaller language data while there is more data 
    // for the other language
    all_train_words = max_train_words * NUM_EPOCHS * NUM_LANG; 
  else {
    all_train_words = EARLY_STOP;
  }
  if (dump_every < 0) {
    dump_every = max_train_words / abs(dump_every);
  }
  fseek(fi, file_sizes[lang_id] / (long long)num_threads * thread_id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  (epoch %lld) Updates (L1: %.2fM, "
            "L2: %.2fM) L1L2grad: %.4f Words/sec: %.2fK  ", 
         13, 
         alpha,
         word_count_actual / (real)(all_train_words + 1) * 100,
		     epoch[0],
         lang_updates[0] / (real)1000000,
         lang_updates[1] / (real)1000000,
         bilbowa_grad,
         word_count_actual / ((real)(now - start + 1) / 
           (real)CLOCKS_PER_SEC * 1000));
         fflush(stdout);
      }
      if (!adagrad) {
        if (word_count_actual < (all_train_words + 1)) {
		      alpha = starting_alpha * 
            (1.0 - word_count_actual / (real)(all_train_words + 1));
        } else alpha = starting_alpha * 0.0001;
      	//if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
	    }
    }
    if (sentence_length == 0) {
      sentence_length = ReadSent(fi, lang_id, mono_sen, 1);
      word_count += sentence_length;
      sentence_position = 0;
    }
    if (lang_updates[lang_id] > all_train_words / NUM_LANG) break;
    if (lang_updates[lang_id] > 0 &&
        lang_updates[lang_id] % max_train_words == 0) {
      epoch[lang_id]++;
    }
    if (feof(fi) || (word_count > train_words[lang_id] / num_threads)) {
      word_count_actual += word_count - last_word_count;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_sizes[lang_id] / (long long)num_threads * thread_id, SEEK_SET);
	    continue;
    }
    if (EARLY_STOP) {
	    if (word_count_actual > EARLY_STOP) {
        fprintf(stderr, "EARLY STOP point reached (thread %d)\n", (int)id);
        break;
	  }
    }
    word = mono_sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    // CBOW ARCHITECTURE WITH NEGATIVE SAMPLING
    if (cbow) {
      for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = tables[lang_id][(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        // learning rate alpha is applied in UpdateEmbeddings()
        if (f > MAX_EXP) g = (label - 1);
        else if (f < -MAX_EXP) g = (label - 0);
        else g = (label - expTable[(int)((f + MAX_EXP) * 
              (EXP_TABLE_SIZE / MAX_EXP / 2))]);
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        //for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
		    for (c = 0; c < layer1_size; c++) syn1negDelta[c] = neu1[c] * g;
    		UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2, layer1_size, 
            syn1negDelta, +1);
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) 
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = mono_sen[c];
          if (last_word == -1) continue;
          //for (c = 0; c < layer1_size; c++) 
          //syn0[c + last_word * layer1_size] += neu1e[c];
          UpdateEmbeddings(syn0, syn0grads[lang_id], last_word * layer1_size, 
              layer1_size, neu1e, +1);
      }
    } else {        
      // SKIPGRAM ARCHITECTURE WITH NEGATIVE SAMPLING
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = mono_sen[c];
          if (last_word == -1) continue;
          l1 = last_word * layer1_size;
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          // NEGATIVE SAMPLING
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = tables[lang_id][(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) 
              f += syn0[c + l1] * syn1neg[c + l2];
            // We multiply with the learning rate in UpdateEmbeddings()
            if (f > MAX_EXP) g = (label - 1);
            else if (f < -MAX_EXP) g = (label - 0);
            else g = (label - expTable[(int)((f + MAX_EXP) *
                  (EXP_TABLE_SIZE / MAX_EXP / 2))]);
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            //for (c = 0; c < layer1_size; c++) 
            //syn1neg[c + l2] += g * syn0[c + l1];
     		    for (c = 0; c < layer1_size; c++) 
              syn1negDelta[c] = g * syn0[c + l1];
		        UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2, layer1_size, 
                syn1negDelta, +1);
          }
          // Learn weights input -> hidden
          //for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
		      UpdateEmbeddings(syn0, syn0grads[lang_id], l1, 
              layer1_size, neu1e, +1);
        }
      }   // for
    }   // skipgram
    lang_updates[lang_id]++;
    sentence_position++;
	  if (dump_every > 0) {
	    if (lang_updates[lang_id] % dump_every == 0) {
	  	  char save_name[MAX_STRING];
	  	  sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++);	
	  	  SaveModel(lang_id, save_name);
	    }
	  }
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  MONO_DONE_TRAINING++;
  pthread_exit(NULL);
}

void TrainModel() {
  long a;
  int lang_id, i;
  pthread_t *pt = malloc((NUM_LANG + 1) * num_threads * sizeof(pthread_t));
  starting_alpha = alpha;
  expTable = malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    // Precompute the exp() table
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    // Precompute sigmoid f(x) = x / (x + 1)
    expTable[i] = expTable[i] / (expTable[i] + 1);                   
  }
  //TODO: CHANGE THIS FOR MORE THAN 2 LANGUAGES
  max_train_words = 0;
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
    vocabs[lang_id] = calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hashes[lang_id] = calloc(vocab_hash_size, sizeof(int));
    if (read_vocab_files[lang_id][0] != 0) {
      fprintf(stderr, "Reading vocab\n");
      ReadVocab(lang_id);
    } else {
      fprintf(stderr, "Learning Vocab\n");
      LearnVocabFromTrainFile(lang_id);
      fprintf(stderr, "Done learning vocab\n");
    }
    if (save_vocab_files[lang_id][0] != 0) {
      fprintf(stderr, "Saving vocab\n");
      SaveVocab(lang_id);
    }
    if (!learn_vocab_and_quit && output_files[lang_id][0] == 0) {
      printf("ERROR: No output name specified.");
      exit(1);
    }
    fprintf(stderr, "Initializing net..");
    InitNet(lang_id);
    fprintf(stderr, "..done.\n");
    fprintf(stderr, "Initializing unigram table..");
    InitUnigramTable(lang_id);
    fprintf(stderr, "..done.\n");
    if (train_words[lang_id] > max_train_words) 
      max_train_words = train_words[lang_id];
  }
  if (learn_vocab_and_quit) exit(0);
  start = clock();
  fprintf(stderr, "Starting training.\n");
  for (a = 0; a < NUM_LANG * num_threads; a++) {
    if (debug_mode > 2) printf("Spawning mono thread %ld\n", a);
    pthread_create(&pt[a], NULL, MonoModelThread, (void *)a);
  }
  for (a = 0; a < (NUM_LANG - 1) * num_threads; a++) {
    if (debug_mode > 2) printf("Spawning parallel thread %ld\n", a);
    pthread_create(&pt[a], NULL, BilbowaThread, (void *)a);
  }
  for (a = 0; a < (NUM_LANG + 1) * num_threads; a++) pthread_join(pt[a], NULL);
  // Save the word vectors
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
    if (dump_iters[lang_id] == 0) { // if user didn't specify -dump_every
      char save_name[MAX_STRING];
      sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++);
      SaveModel(lang_id, save_name);
    }
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i, lang_id;
  if (argc == 1) {
    printf("Bilingual Bag-Of-Words without Alignments (BilBOWA) cross-lingual word "
        "vector estimation toolkit v0.1a\n\n");
    printf("Options:\n");
    printf("Arguments for training:\n");
    printf("\t-mono-trainN <file>\n");
    printf("\t\tUse monolingual text data for language N from <file> to train\n");
    printf("\t-par-trainN <file>\n");
    printf("\t\tUse parallel text data for language N from <file> to train\n");
    printf("\t-outputN <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors for language N\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with"
        " higher frequency\n");
    printf(" in the training data will be randomly down-sampled; default is "
        "0 (off), useful value is 1e-5\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are"
        " 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; "
        "default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2, more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary mode; default is 0 (off)\n");
    printf("\t-save-vocabN <file>\n");
    printf("\t\tThe vocabulary for language N will be saved to <file>\n");
    printf("\t-read-vocabN <file>\n");
    printf("\t\tThe vocabulary for language N will be read from <file>, not "
        "constructed from the training data\n");
    printf("\t-epochs N\n");
    printf("\t\tTrain for N epochs (default = 1)\n");
    printf("\t-adagrad <int>\n");
    printf("\t\tUse Adagrad adaptive learning rate anealing (default = 1)\n");
    printf("\t-xling-lambda <float>\n");
    printf("\t\tCross-lingual regularization weight\n");
    printf("\t-dump-every N\n");
    printf("\t\tSave intermediate embeddings during training every N steps if N>0,"
        " else every epoch/N steps\n");
    printf("\t-plusplus <int>\n");
    printf("\t\tUse sentence-vector model (0, default) or use diagonal alignment"
        " model (1)\n");
    printf("\t-align-lambda <float>\n");
    printf("\t\tWeight controlling how strongly it is preferred to align words "
        "along the sentence-diagonal\n");
    printf("\t-lean-vocab-and-quit <int>\n");
    printf("\t\tLearn and save vocab only\n");
    printf("\nExamples:\n");
    printf("./bilbowa -mono-train1 endata.txt -mono-train2 frdata.txt -par-train1 "
       "enfr.en -par-train2 enfr.fr -output1 envec.txt -output2 frvec.txt -size 100"
       " -window 5 -sample 1e-4 -negative 5 -binary 0 -adagrad 1 -plusplus 0\n\n");
    return 0;
  }

  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
    mono_train_files[lang_id] = calloc(MAX_STRING, sizeof(char));
    par_train_files[lang_id] = calloc(MAX_STRING, sizeof(char));
    output_files[lang_id] = calloc(MAX_STRING, sizeof(char));
    save_vocab_files[lang_id] = calloc(MAX_STRING, sizeof(char));
    read_vocab_files[lang_id] = calloc(MAX_STRING, sizeof(char));
    lang_updates[lang_id] = 0;
	  dump_iters[lang_id] = 0;
  }
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) 
    layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train1", argc, argv)) > 0) 
    strcpy(mono_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train2", argc, argv)) > 0) 
    strcpy(mono_train_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train1", argc, argv)) > 0) 
    strcpy(par_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train2", argc, argv)) > 0) 
    strcpy(par_train_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab1", argc, argv)) > 0) 
    strcpy(save_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab1", argc, argv)) > 0) 
    strcpy(read_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab2", argc, argv)) > 0) 
    strcpy(save_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab2", argc, argv)) > 0) 
    strcpy(read_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) 
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) 
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) 
    alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output1", argc, argv)) > 0) 
    strcpy(output_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-output2", argc, argv)) > 0) 
    strcpy(output_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) 
    window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) 
    sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) 
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) 
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) 
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-early-stop", argc, argv)) > 0) 
    EARLY_STOP = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) 
    NUM_EPOCHS = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-adagrad", argc, argv)) > 0)
    adagrad = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-xling-lambda", argc, argv)) > 0)
    XLING_LAMBDA = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-dump-every", argc, argv)) > 0) 
    dump_every = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-plusplus", argc, argv)) > 0) 
    PLUSPLUS = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-align-lambda", argc, argv)) > 0) 
    ALIGN_LAMBDA = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-learn-vocab-and-quit", argc, argv)) > 0) 
    learn_vocab_and_quit = atoi(argv[i + 1]);

  TrainModel();
  return 0;
}
