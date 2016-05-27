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
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name_lang1[max_size], file_name_lang2[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words_lang1, words_lang2, size_lang1, size_lang2, a, b, c, d, cn, bi[100];
  char ch;
  float *M_lang1, *M_lang2;
  char *vocab_lang1, *vocab_lang2;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name_lang1, argv[1]);
  f = fopen(file_name_lang1, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words_lang1);
  fscanf(f, "%lld", &size_lang1);
  vocab_lang1 = (char *)malloc((long long)words_lang1 * max_w * sizeof(char));
  M_lang1 = (float *)malloc((long long)words_lang1 * (long long)size_lang1 * sizeof(float));
  if (M_lang1 == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words_lang1 * size_lang1 * sizeof(float) / 1048576, words_lang1, size_lang1);
    return -1;
  }
  for (b = 0; b < words_lang1; b++) {
    fscanf(f, "%s%c", &vocab_lang1[b * max_w], &ch);
    for (a = 0; a < size_lang1; a++) fread(&M_lang1[a + b * size_lang1], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size_lang1; a++) len += M_lang1[a + b * size_lang1] * M_lang1[a + b * size_lang1];
    len = sqrt(len);
    for (a = 0; a < size_lang1; a++) M_lang1[a + b * size_lang1] /= len;
  }
  fclose(f);

//////// Load Languag 2
  strcpy(file_name_lang2, argv[2]);
  f = fopen(file_name_lang2, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words_lang2);
  fscanf(f, "%lld", &size_lang2);
  vocab_lang2 = (char *)malloc((long long)words_lang2 * max_w * sizeof(char));
  M_lang2 = (float *)malloc((long long)words_lang2 * (long long)size_lang2 * sizeof(float));
  if (M_lang2 == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words_lang2 * size_lang2 * sizeof(float) / 1048576, words_lang2, size_lang2);
    return -1;
  }
  for (b = 0; b < words_lang2; b++) {
    fscanf(f, "%s%c", &vocab_lang2[b * max_w], &ch);
    for (a = 0; a < size_lang2; a++) fread(&M_lang2[a + b * size_lang2], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size_lang2; a++) len += M_lang2[a + b * size_lang2] * M_lang2[a + b * size_lang2];
    len = sqrt(len);
    for (a = 0; a < size_lang2; a++) M_lang2[a + b * size_lang2] /= len;
  }
  fclose(f);


  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter a word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words_lang1; b++) if (!strcmp(&vocab_lang1[b * max_w], st[a])) break;
      if (b == words_lang1) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size_lang2; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size_lang2; a++) vec[a] += M_lang2[a + bi[b] * size_lang2];
    }
    len = 0;
    for (a = 0; a < size_lang2; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size_lang2; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words_lang2; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      //for (a = 0; a < size_lang1; a++) dist += vec[a] * M_lang1[a + c * size_lang1];
      for (a = 0; a < size_lang2; a++) dist += vec[a] * M_lang2[a + c * size_lang2];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          //strcpy(bestw[a], &vocab_lang1[c * max_w]);
          strcpy(bestw[a], &vocab_lang2[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
  }
  return 0;
}
