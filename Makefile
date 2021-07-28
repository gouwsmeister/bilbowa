bilbowa: src/bilbowa.v0.1.c 
	mkdir -p bin
	gcc -std=c99 -Wall -O2 -march=native -funroll-loops src/bilbowa.v0.1.c -o bin/bilbowa -lm -pthread
bidist: src/bidist.c
	gcc -std=c99 -Wall -O2 -march=native -funroll-loops src/bidist.c -o bin/bidist -lm
all: bilbowa bidist
clean:
	rm -fv bin/*
