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
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define NUM_LANG 2

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char *mono_train_files[NUM_LANG],
	 *par_train_files[NUM_LANG], 
	 *noise_train_files[NUM_LANG - 1],
	 *output_files[NUM_LANG], 
     *save_vocab_files[NUM_LANG], 
     *read_vocab_files[NUM_LANG];
struct vocab_word *vocabs[NUM_LANG];
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, 
    num_threads = NUM_LANG, min_reduce = 1;
int *vocab_hashes[NUM_LANG];
long long vocab_max_size = 1000, vocab_sizes[NUM_LANG], layer1_size = 40;
long long train_words[NUM_LANG], word_count_actual, file_sizes[NUM_LANG];
long long lang_updates[NUM_LANG], dump_every=0, dump_iters[NUM_LANG];
unsigned long long next_random = 0;
real alpha = 0.025, starting_alpha, sample = 0, adagrad = 1;
real *syn0s[NUM_LANG], *syn1s[NUM_LANG], *syn1negs[NUM_LANG], 
	 *syn0grads[NUM_LANG], *syn1negGrads[NUM_LANG], *expTable;
clock_t start;

const int table_size = 1e8;     // const across languages
int *tables[NUM_LANG];
int negative = 15, MONO_DONE_TRAINING = 0;
long long NUM_EPOCHS=1, EARLY_STOP = 0, max_train_words;
real *delta_pos, XLING_LAMBDA = 0.1, MARGIN = 0;

void InitUnigramTable(int lang_id) {
  int a, i;
  long long train_words_pow = 0, vocab_size = vocab_sizes[lang_id];
  int *table;
  struct vocab_word *vocab = vocabs[lang_id];
  real d1, power = 0.75;
  table = tables[lang_id] = (int *)malloc(table_size * sizeof(int));
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
  vocab[vocab_sizes[lang_id]].word = (char *)calloc(length, sizeof(char));
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
  qsort(&vocab[1], vocab_sizes[lang_id] - 1, sizeof(struct vocab_word), VocabCompare);
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
  vocabs[lang_id] = (struct vocab_word *)realloc(vocabs[lang_id], (vocab_sizes[lang_id] + 1) * sizeof(struct vocab_word));
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
    printf("ERROR: training data (%s) file not found (lang_id==%d)!\n", train_file, lang_id);
    exit(1);
  }
  vocab_sizes[lang_id] = 0;
  AddWordToVocab(lang_id, (char *)"</s>");
  vocab = vocabs[lang_id];
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words[lang_id]++;
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
  printf("Saving vocabulary with %lld entries to %s\n", vocab_sizes[lang_id], save_vocab_file);
  for (i = 0; i < vocab_sizes[lang_id]; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
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
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn0s[lang_id] = syn0;
  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn1negs[lang_id] = syn1neg;
  if (adagrad) {
	a = posix_memalign((void **)&syn0grad, 128, (long long)vocab_size * layer1_size * sizeof(real));
  	if (syn0grad == NULL) {printf("Memory allocation failed\n"); exit(1);}
  	else syn0grads[lang_id] = syn0grad;
  	a = posix_memalign((void **)&syn1negGrad, 128, (long long)vocab_size * layer1_size * sizeof(real));
  	if (syn1negGrad == NULL) {printf("Memory allocation failed\n"); exit(1);}
	else syn1negGrads[lang_id] = syn1negGrad;
  }
  for (b = 0; b < layer1_size; b++) 
    for (a = 0; a < vocab_size; a++) {
      syn1neg[a * layer1_size + b] = 0;
      syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	  if (adagrad) {
		syn0grad[a*layer1_size + b] = 0;
	    syn1negGrad[a*layer1_size + b] = 0;
	  }
    }
}

/* Read a sentence into *sen using vocabulary for language lang_id
 * Store processed words in *word_count, returns (potentially subsampled) 
 * length of sentence */
int ReadSent(FILE *fi, int lang_id, long long *sen, long long *word_count) {
  long long word;
  int sentence_length = 0;
  struct vocab_word *vocab = vocabs[lang_id];

  while (1) {
    word = ReadWordIndex(fi, lang_id);
    if (feof(fi)) break;
    if (word == -1) continue;       // unknown
    if (word_count) (*word_count)++;
    if (word == 0) break;           // end-of-sentence
    // The subsampling randomly discards frequent words while keeping the 
    // ranking the same
    if (sample > 0) {
      real ran = (sqrt(vocab[word].cn / (sample * train_words[lang_id])) + 1) * (sample * train_words[lang_id]) / vocab[word].cn;
      next_random = next_random * (unsigned long long)25214903917 + 11;
      if (ran < (next_random & 0xFFFF) / (real)65536) continue;
    }
    sen[sentence_length] = word;
    sentence_length++;
    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
  }
  return sentence_length;
}

void UpdateEmbeddings(real *embeddings, real *grads, int offset, 
					  int num_updates, real *deltas, real weight) {
  int a;
  real step, fudge_factor = 1.0;
  for (a = 0; a < num_updates; a++) {
	// We use Adagrad for automatic learning rate selection
	if (adagrad) {
	  grads[offset + a] += (deltas[a] * deltas[a]);
	  step = alpha * deltas[a] / (fudge_factor + sqrt(grads[offset + a]));
	} else {
	  step = alpha * deltas[a];
	}
	if (step != step) {
	  fprintf(stderr, "ERROR: step == NaN\n");
	}
    // TODO: Clip updates to [-0.1, +0.1]
	embeddings[offset + a] += weight * step;   
  }
}

void UpdateEnFrSquaredError(int en_sen_len, int fr_sen_len, 
                			long long *en_sen, long long *fr_sen, 
                			real *delta, real *syn0_e, real *syn0_f, 
                            real weight) {
  int d, offset;
  // TO MINIMIZE SQUARED ERROR:
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

real fpropSent(int len, long long *sen, real *delta, real *syn, real sign) {
  real sumSquares = 0;
  long long c, d, offset;
  for (d = 0; d < len; d++) {
     offset = layer1_size * sen[d];
     for (c = 0; c < layer1_size; c++) {
       delta[c] += sign * syn[offset + c];    
       sumSquares += delta[c] * delta[c];
     }
  }
  return sqrt(sumSquares);
}

/* Thread for performing the cross-lingual learning:
 * Supports both using only an L2 penalty on observed pairs (negative = 0),
 * and a negative hinge loss on observed and sampled noise pairs (negative > 0).
 * */
void *BilbowaUpdateThread(void *id) {
    int par_sen_len1, par_sen_len2, 
		lang_id1 = (int)id, lang_id2 = (int)id + 1;
    long long par_sen1[MAX_SENTENCE_LENGTH], 
              par_sen2[MAX_SENTENCE_LENGTH],
			  updates_l1 = 0, updates_l2 = 0;
    int a;
    real *syn0_e = syn0s[lang_id1], *syn0_f = syn0s[lang_id2];
    real enfr_delta[layer1_size], xling_balancer;
    FILE *fi_par1, *fi_par2;
      
    fi_par1 = fopen(par_train_files[lang_id1], "rb");   // en
    fi_par2 = fopen(par_train_files[lang_id2], "rb");   // fr 

	// Continue training while skipgrams are still training
    while (MONO_DONE_TRAINING < NUM_LANG) {
      par_sen_len1 = ReadSent(fi_par1, lang_id1, par_sen1, &updates_l1);
      par_sen_len2 = ReadSent(fi_par2, lang_id2, par_sen2, &updates_l2);
      if (feof(fi_par1) || feof(fi_par2)) {
          fseek(fi_par1, 0, SEEK_SET);       // recycle parallel sentences
          fseek(fi_par2, 0, SEEK_SET);
      }
      // FPROP
      // RESET DELTAS
      for (a = 0; a < layer1_size; a++) enfr_delta[a] = 0;
      // ACCUMULATE L2 LOSS DELTA
	  // TODO: Use syn0 / syn1neg?
      fpropSent(par_sen_len1, par_sen1, enfr_delta, syn0_e, +1); 
      fpropSent(par_sen_len2, par_sen2, enfr_delta, syn0_f, -1);
      // BPROP
	  xling_balancer = (lang_updates[0] + lang_updates[1]) / (real)(updates_l1 + updates_l2);
      UpdateEnFrSquaredError(par_sen_len1, par_sen_len2, 
							 par_sen1, par_sen2,
							 enfr_delta, syn0_e, syn0_f, XLING_LAMBDA * xling_balancer);
    } // while
  fclose(fi_par1);
  fclose(fi_par2);
  pthread_exit(NULL);
}


/* Thread for performing the cross-lingual learning (L2 loss)
void *BilbowaUpdateThread(void *id) {
    int par_sen_len1, par_sen_len2, lang_id1 = (int)id, lang_id2 = (int)id + 1;
    long long par_sen1[MAX_SENTENCE_LENGTH], par_sen2[MAX_SENTENCE_LENGTH];
    int a, c, d, offset;
    real *syn0_e = syn0s[lang_id1], *syn0_f = syn0s[lang_id2];
    real xling_delta[layer1_size], step;
    FILE *fi_par1, *fi_par2;
      
    fi_par1 = fopen(par_train_files[lang_id1], "rb");
    fi_par2 = fopen(par_train_files[lang_id2], "rb");

    while (MONO_DONE_TRAINING < NUM_LANG) {
      par_sen_len1 = ReadSent(fi_par1, lang_id1, par_sen1, NULL);
      par_sen_len2 = ReadSent(fi_par2, lang_id2, par_sen2, NULL);
      if (feof(fi_par1) || feof(fi_par2)) {
          fseek(fi_par1, 0, SEEK_SET);       // recycle parallel sentences
          fseek(fi_par2, 0, SEEK_SET);
      }
      // RESET ERROR
      for (a = 0; a < layer1_size; a++) xling_delta[a] = 0;
      // ACCUMULATE ERROR
      for (d = 0; d < par_sen_len1; d++) {
         offset = layer1_size * par_sen1[d];
         for (c = 0; c < layer1_size; c++) {
           xling_delta[c] += syn0_e[offset + c];    // TODO: syn0 / syn1neg?
         }
      }
      for (d = 0; d < par_sen_len2; d++) {
        offset = layer1_size * par_sen2[d];
        for (c = 0; c < layer1_size; c++) {
          xling_delta[c] -= syn0_f[offset + c];
        }
      }
      // UPDATE PARAMETERS
      // delta = d .5*|| e - f ||^2 = (e - f)
      // TO MINIMIZE SQUARED ERROR:
      // d/den = delta 
      for (d = 0; d < par_sen_len1; d++) {
        offset = layer1_size * par_sen1[d];
        for (a = 0; a < layer1_size; a++) {
          step = XLING_LAMBDA * alpha * xling_delta[a];
          if (step != step) {
            fprintf(stderr, "ERROR: step == NaN\n");
            sleep(1);
          }
          // update in -d/den direction
          syn0_e[offset + a] -= step;           // TODO: syn0 / syn1neg ??
        }
      }
      // d/df = -delta
      for (d = 0; d < par_sen_len2; d++) {
        offset = layer1_size * par_sen2[d];
        for (a = 0; a < layer1_size; a++) {
          step = XLING_LAMBDA * alpha * xling_delta[a];
          if (step != step) {
            fprintf(stderr, "ERROR: step == NaN\n");
            sleep(1);
          }
          // update in -d/dfr direction
           syn0_f[offset + a] += step;
        }
      }
    } //while
  fclose(fi_par1);
  fclose(fi_par2);
  pthread_exit(NULL);
}

*/

void SaveModel(int lang_id, char *name) {
  long a, b;
  struct vocab_word *vocab = vocabs[lang_id];
  real *syn0 = syn0s[lang_id];
  FILE *fo = fopen(name, "wb");

  printf("Saving model to file: %s\n", name);
  fprintf(fo, "%lld %lld\n", vocab_sizes[lang_id], layer1_size);
  for (a = 0; a < vocab_sizes[lang_id]; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

/* Monolingual training thread */ 
void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, all_train_words = 0, epoch=0;
  long long mono_sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  int lang_id = (int)id;       // TODO: Fix this
  char *train_file = mono_train_files[lang_id];
  //struct vocab_word *vocab = vocabs[lang_id];
  long long vocab_size = vocab_sizes[lang_id];
  //unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  real *syn1neg = syn1negs[lang_id];
  real *syn1negDelta = (real *)calloc(layer1_size, sizeof(real));
  real *syn0 = syn0s[lang_id];
  FILE *fi = fopen(train_file, "rb");

  if (dump_every == -1) dump_every = max_train_words;		// every epoch
  if (!EARLY_STOP)
    // If two languages have different amounts of training data,
    // recycle the smaller language data while there is more data 
    // for the other language
    all_train_words = max_train_words * NUM_LANG * NUM_EPOCHS; 
  else {
    all_train_words = EARLY_STOP;
  }
  // TODO: Fix seeking to take into account NUM_LANG and thread_id
  //fseek(fi, file_sizes[lang_id] / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  (epoch %lld) Updates (L1: %lld, L2: %lld) Words/sec: %.2fk  ", 
         13, 
         alpha,
         word_count_actual / (real)(all_train_words + 1) * 100,
		 epoch,
         lang_updates[0],
         lang_updates[1],
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }`
      if (!adagrad) {
		alpha = starting_alpha * (1 - word_count_actual / (real)(all_train_words + 1));
      	if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
	  }
    }
    if (sentence_length == 0) {
      sentence_length = ReadSent(fi, lang_id, mono_sen, &word_count);
      sentence_position = 0;
    }
    // TODO: take num_threads/EPOCHS into account below
    if (word_count_actual > all_train_words) break;
    if (feof(fi)) {
	  epoch++;
      fseek(fi, 0, SEEK_SET);
	  if (dump_every > 0) {
		// save every epoch
 	  	char save_name[MAX_STRING];
	  	sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++);	
	  	SaveModel(lang_id, save_name);
	  }
 	  continue;
    }
    if (EARLY_STOP) {
	  if (word_count_actual > EARLY_STOP) {
        //thread_pool[(int)id] = 'Q';       // thread quit
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
    // CBOW ARCHITECTURE: NEGATIVE SAMPLING
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
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        //for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
		for (c = 0; c < layer1_size; c++) syn1negDelta[c] = neu1[c] * g;
		UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2, layer1_size, syn1negDelta, +1);
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = mono_sen[c];
        if (last_word == -1) continue;
        //for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
		UpdateEmbeddings(syn0, syn0grads[lang_id], last_word*layer1_size, layer1_size, neu1e, +1);
      }
    } else {        
      // SKIPGRAM ARCHITECTURE: NEGATIVE SAMPLING
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
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            //for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
     		for (c = 0; c < layer1_size; c++) syn1negDelta[c] = g * syn0[c + l1];
			UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2, layer1_size, syn1negDelta, +1);
          }
          // Learn weights input -> hidden
          //for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
		  UpdateEmbeddings(syn0, syn0grads[lang_id], l1, layer1_size, neu1e, +1);
        }
      }   // for
    }   // skipgram
    lang_updates[lang_id]++;
    sentence_position++;
	//if (dump_every > 0) {
	//  if ((word_count - sentence_position) % dump_every == 0) {
	//	char save_name[MAX_STRING];
	//	sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++);	
	//	SaveModel(lang_id, save_name);
	//  }
	//}
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
  pthread_t *pt = (pthread_t *)malloc((NUM_LANG + 1) * sizeof(pthread_t));

  starting_alpha = alpha;
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    // Precompute the exp() table
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    // Precompute sigmoid f(x) = x / (x + 1)
    expTable[i] = expTable[i] / (expTable[i] + 1);                   
  }
  //TODO: CHANGE THIS FOR MORE THAN 2 LANGUAGES
  max_train_words = 0;
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
    vocabs[lang_id] = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hashes[lang_id] = (int *)calloc(vocab_hash_size, sizeof(int));
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
    if (output_files[lang_id][0] == 0) return;
    fprintf(stderr, "Initializing net..");
    InitNet(lang_id);
    fprintf(stderr, "..done.\n");
    fprintf(stderr, "Initializing unigram table..");
    InitUnigramTable(lang_id);
    fprintf(stderr, "..done.\n");
    if (train_words[lang_id] > max_train_words) 
      max_train_words = train_words[lang_id];
  }
  start = clock();
  fprintf(stderr, "Starting training.\n");
  // TODO: Change this to support multiple langs and threads per language
  //num_threads = NUM_LANG + 1;       
  for (a = 0; a < NUM_LANG; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  pthread_create(&pt[a], NULL, BilbowaUpdateThread, (void *)0);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  // Save the word vectors
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
	char save_name[MAX_STRING];
	sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++);
    SaveModel(lang_id, save_name);
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
    printf("BILINGUAL BOW w/out Alignments (BilBOWA) WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-par-trainN <file>\n");
    printf("\t\tUse parallel text data for language N from <file> to train the model\n");
    printf("\t-outputN <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors for language N\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocabN <file>\n");
    printf("\t\tThe vocabulary for language N will be saved to <file>\n");
    printf("\t-read-vocabN <file>\n");
    printf("\t\tThe vocabulary for language N will be read from <file>, not constructed from the training data\n");
   printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
    mono_train_files[lang_id] = (char *)calloc(MAX_STRING, sizeof(char));
    par_train_files[lang_id] = (char *)calloc(MAX_STRING, sizeof(char));
    output_files[lang_id] = (char *)calloc(MAX_STRING, sizeof(char));
    save_vocab_files[lang_id] = (char *)calloc(MAX_STRING, sizeof(char));
    read_vocab_files[lang_id] = (char *)calloc(MAX_STRING, sizeof(char));
    lang_updates[lang_id] = 0;
	dump_iters[lang_id] = 0;
  }
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train1", argc, argv)) > 0) strcpy(mono_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train2", argc, argv)) > 0) strcpy(mono_train_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train1", argc, argv)) > 0) strcpy(par_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train2", argc, argv)) > 0) strcpy(par_train_files[1], argv[i + 1]);
  //TODO: make this support more than one language pair
  if ((i = ArgPos((char *)"-noise-train", argc, argv)) > 0) strcpy(noise_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab1", argc, argv)) > 0) strcpy(save_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab1", argc, argv)) > 0) strcpy(read_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab2", argc, argv)) > 0) strcpy(save_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab2", argc, argv)) > 0) strcpy(read_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) MARGIN = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output1", argc, argv)) > 0) strcpy(output_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-output2", argc, argv)) > 0) strcpy(output_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-early-stop", argc, argv)) > 0) EARLY_STOP = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) NUM_EPOCHS = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-adagrad", argc, argv)) > 0) adagrad = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-xling-lambda", argc, argv)) > 0) XLING_LAMBDA = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-dump-every", argc, argv)) > 0) dump_every = atoi(argv[i + 1]);
  
  if (MARGIN == 0) MARGIN = layer1_size;
  TrainModel();
  return 0;
}
