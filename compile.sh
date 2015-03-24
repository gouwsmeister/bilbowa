#! /bin/bash

mkdir -p bin

#gcc src/bilbowa++.v0.1.c -g -o bin/bilbowa++ -lm -pthread -Ofast -march=native -Wall -funroll-loops
gcc src/bilbowa.v0.1.c -g -o bin/bilbowa -lm -pthread -Ofast -march=native -Wall -funroll-loops

