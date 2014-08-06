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
     *output_files[NUM_LANG], 
     *save_vocab_files[NUM_LANG], 
     *read_vocab_files[NUM_LANG];
struct vocab_word *vocabs[NUM_LANG];
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = NUM_LANG, min_reduce = 1;
int *vocab_hashes[NUM_LANG];
long long vocab_max_size = 1000, vocab_sizes[NUM_LANG], layer1_size = 1000;
long long train_words[NUM_LANG], word_count_actuals, file_sizes[NUM_LANG];
long long lang_updates[NUM_LANG];
unsigned long long next_random = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0s[NUM_LANG], *syn1s[NUM_LANG], *syn1negs[NUM_LANG], *expTable;
clock_t start;

const int table_size = 1e8;     // const across languages
int *tables[NUM_LANG];
int negative = 15;
long long EARLY_STOP = 0;

int XLING_WAIT = 0, waiting_for_xling = 0;
real *xling_delta, XLING_LAMBDA = 1.0;
char thread_pool[NUM_LANG];

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

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
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

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
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

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, int lang_id) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(lang_id, word);         // MOD
}


// Adds a word to the vocabulary
int AddWordToVocab(int lang_id, char *word) {    
  unsigned int hash, length = strlen(word) + 1;
  struct vocab_word *vocab = vocabs[lang_id];
  //long long vocab_size = vocab_sizes[lang_id]; 		// array of long longs
  int *vocab_hash = vocab_hashes[lang_id];			// array of *ints
  
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_sizes[lang_id]].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_sizes[lang_id]].word, word);
  vocab[vocab_sizes[lang_id]].cn = 0;
  vocab_sizes[lang_id]++;
  // Reallocate memory if needed
  if (vocab_sizes[lang_id] + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocabs[lang_id] = (struct vocab_word *)realloc(vocabs[lang_id], vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_sizes[lang_id] - 1;
  return vocab_sizes[lang_id] - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *word1, const void *word2) {
    return ((struct vocab_word *)word2)->cn - ((struct vocab_word *)word1)->cn;
}

// Sorts the vocabulary by frequency using word counts
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
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words[lang_id] += vocab[a].cn;
    }
  }
  vocabs[lang_id] = (struct vocab_word *)realloc(vocabs[lang_id], (vocab_sizes[lang_id] + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  //for (a = 0; a < vocab_size; a++) {
  //  vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
  //  vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  //}
}

// Reduces the vocabulary by removing infrequent tokens
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
    // Hash will be re-computed, as it is not actual
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
      //fprintf(stderr, "pre-AddWordtoVocab\n");
      a = AddWordToVocab(lang_id, word);
      vocab = vocabs[lang_id];      // might have changed
      //fprintf(stderr, "post-AddWordToVocab\n");
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_sizes[lang_id] > vocab_hash_size * 0.7) {
	  ReduceVocab(lang_id);     // MOD
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
  //struct vocab_word *vocab = vocabs[lang_id];
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
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_sizes[lang_id] = ftell(fin);
  fclose(fin);
}

void InitNet(int lang_id) {
  long long a, b, vocab_size = vocab_sizes[lang_id];
  real *syn0, *syn1neg;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn0s[lang_id] = syn0;
  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  else syn1negs[lang_id] = syn1neg;
  for (b = 0; b < layer1_size; b++) 
    for (a = 0; a < vocab_size; a++) {
      syn1neg[a * layer1_size + b] = 0;
      syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    }
}

// Read a sentence into *sen using vocabulary for language lang_id
// Store processed words in *word_count, returns 
// (potentially subsampled) length of sentence
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
    // The subsampling randomly discards frequent words while keeping the ranking
    // the same
    if (sample > 0) {
      real ran = (sqrt(vocab[word].cn / (sample * train_words[lang_id])) + 1) * 
        (sample * train_words[lang_id]) / vocab[word].cn;
      next_random = next_random * (unsigned long long)25214903917 + 11;
      if (ran < (next_random & 0xFFFF) / (real)65536) continue;
    }
    sen[sentence_length] = word;
    sentence_length++;
    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
  }
  return sentence_length;
}

void BilbowaUpdate() {

}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, all_train_words = 0;
  long long mono_sen[MAX_SENTENCE_LENGTH + 1], par_sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, offset, target, label;
  int lang_id = (int)id, xling_sign, par_sen_len;       // TODO: Fix this
  char *train_file = mono_train_files[lang_id], NaN;
  //struct vocab_word *vocab = vocabs[lang_id];
  long long vocab_size = vocab_sizes[lang_id];
  //unsigned long long next_random = (long long)id;
  real f, g, step;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  real *syn1neg = syn1negs[lang_id];
  real *syn0 = syn0s[lang_id];
  FILE *fi = fopen(train_file, "rb");
  FILE *fi_par = fopen(par_train_files[lang_id], "rb");
  // this compute the sign of the gradient on the xling layer, used below
  if (lang_id % 2) xling_sign = +1;   // even ids add
  else xling_sign = -1;               // odd ids subtract

  if (!EARLY_STOP)
    for (a = 0; a < NUM_LANG; a++) all_train_words += train_words[a];
  else {
    all_train_words = EARLY_STOP;
  }
  // TODO: Fix seeking to take into account NUM_LANG and thread_id
  //fseek(fi, file_sizes[lang_id] / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actuals += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Updates (L1: %lld, L2: %lld) Words/sec: %.2fk  ", 
            13, 
            alpha,
            word_count_actuals / (real)(all_train_words + 1) * 100,
            lang_updates[0],
            lang_updates[1],
            word_count_actuals / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actuals / (real)(all_train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      sentence_length = ReadSent(fi, lang_id, mono_sen, &word_count);
      sentence_position = 0;
    }
    if (feof(fi)) break;
    //if (word_count > train_words[lang_id] / num_threads) break;
    if (EARLY_STOP) {
      if (word_count > EARLY_STOP) {
        thread_pool[(int)id] = 'Q';       // quit
        fprintf(stderr, "EARLY STOP point reached (thread %d)\n", (int)id);
        break;
      }
    }
    if (word_count > train_words[lang_id]) break;
    word = mono_sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    // CBOW: NEGATIVE SAMPLING
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
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = mono_sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else {        // SKIPGRAM
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
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
          }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
      }   // for
    }   // skipgram
    // CROSSLINGUAL UPDATE
    par_sen_len = ReadSent(fi_par, lang_id, par_sen, NULL);
    //printf("Lang %d, parallel sentence (%d words): ", lang_id, par_sen_len);
    //for (a = 0; a < par_sen_len; a++)
    //  printf("%lld ", par_sen[a]);
    //printf("\n");
    if (feof(fi_par)) fseek(fi_par, 0, SEEK_SET);       // recycle parallel sentences
    for (d = 0; d < par_sen_len; d++) {
       offset = layer1_size * par_sen[d];
       for (c = 0; c < layer1_size; c++) {
         xling_delta[c] += xling_sign * syn0[offset + c];    // TODO: syn0 / syn1neg?
       }
    }
    // Synchronize xling gradient accumulation
    num_threads = NUM_LANG;                      // TODO: fix this to handle arbitrary languages/threads
    XLING_WAIT = 1;
    thread_pool[(int)id] = 'R';                  // Running
    if (++waiting_for_xling >= NUM_LANG) {       // last thread per lang pair releases lock
      waiting_for_xling = 0;
      XLING_WAIT = 0;
    } else while (XLING_WAIT) {
        thread_pool[(int)id] = 'L'; // locked
        // below I use "!= 'R'" since status can be {'R','L','Q'}
        for (a = 0; a < num_threads && thread_pool[a] != 'R'; a++);   // deadlock workaround
        if (a == num_threads) XLING_WAIT = 0;
    }
    //NaN = 0;
    //for (a = 0; a < layer1_size; a++) {
    //  printf("%.4f ", xling_delta[a]);
    //  if (xling_delta[a] != xling_delta[a]) NaN = 1;
    //}
    //if (NaN) sleep(1);
    printf("\n");
    // do the xling update per language (per thread)
    for (d = 0; d < par_sen_len; d++) {
      offset = layer1_size * par_sen[d];
      for (a = 0; a < layer1_size; a++) {
        step = XLING_LAMBDA * alpha * xling_delta[a];
        //if (step != step) {
        //  fprintf(stderr, "ERROR: step == NaN\n");
        //  sleep(1);
        //}
        syn0[offset + a] += step;           // TODO: syn0 / syn1neg ??
      }
    }
    // Synchronise xling clearing of xling_delta
    // (we must wait until all threads have consumed xling_delta
    // before we can reset it)
    XLING_WAIT = 1;
    thread_pool[(int)id] = 'R';
    if (++waiting_for_xling >= NUM_LANG) {       // last thread per lang pair releases lock
      for (a = 0; a < layer1_size; a++) xling_delta[a] = 0;  // clear deltas
      waiting_for_xling = 0;
      XLING_WAIT = 0;
    } else while (XLING_WAIT) {
        thread_pool[(int)id] = 'L'; // locked
        for (a = 0; (a < num_threads) && (thread_pool[a] != 'R'); a++);   // deadlock workaround
        if (a == num_threads) XLING_WAIT = 0;
    }
    lang_updates[lang_id]++;
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  fclose(fi_par);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void SaveModel(int lang_id) {
  long a, b;
  struct vocab_word *vocab = vocabs[lang_id];
  real *syn0 = syn0s[lang_id];
  FILE *fo = fopen(output_files[lang_id], "wb");

  fprintf(fo, "%lld %lld\n", vocab_sizes[lang_id], layer1_size);
  for (a = 0; a < vocab_sizes[lang_id]; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void TrainModel() {
  long a;
  int lang_id, i;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

  starting_alpha = alpha;
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute sigmoid f(x) = x / (x + 1)
  }
  //TODO: CHANGE THIS FOR MORE THAN 2 LANGUAGES
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
  }
  start = clock();
  fprintf(stderr, "Starting training.\n");
  num_threads = NUM_LANG;       // TODO: Change this to support multiple langs and threads per lang
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  // Save the word vectors
  for (lang_id = 0; lang_id < NUM_LANG; lang_id++)
    SaveModel(lang_id);
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
    printf("\t-mono-trainN <file>\n");
    printf("\t\tUse monolingual text data for language N from <file> to train the model\n");
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
  }
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train1", argc, argv)) > 0) strcpy(mono_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-mono-train2", argc, argv)) > 0) strcpy(mono_train_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train1", argc, argv)) > 0) strcpy(par_train_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-par-train2", argc, argv)) > 0) strcpy(par_train_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab1", argc, argv)) > 0) strcpy(save_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab1", argc, argv)) > 0) strcpy(read_vocab_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab2", argc, argv)) > 0) strcpy(save_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab2", argc, argv)) > 0) strcpy(read_vocab_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output1", argc, argv)) > 0) strcpy(output_files[0], argv[i + 1]);
  if ((i = ArgPos((char *)"-output2", argc, argv)) > 0) strcpy(output_files[1], argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-early-stop", argc, argv)) > 0) EARLY_STOP = atoi(argv[i + 1]);
  
  posix_memalign((void **)&xling_delta, 128, (long long)layer1_size * sizeof(real));
  if (xling_delta == NULL) {
    printf("ERROR allocating xling_delta\n");
    exit(1);
  } else {
    for (i = 0; i < layer1_size; i++)
      xling_delta[i] = 0;
  }
  
  TrainModel();
  return 0;
}
