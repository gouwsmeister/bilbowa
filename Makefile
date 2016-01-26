bilbowa: src/bilbowa.v0.1.c 
	mkdir -p bin
	gcc src/bilbowa.v0.1.c -g -o bin/bilbowa -lm -pthread -march=native -funroll-loops -w 
bidist: src/bidist.c
	gcc src/bidist.c -g -o bin/bidist -lm -march=native -funroll-loops -w
all: bilbowa bidist
clean:
	rm bin/bilbowa bin/bidist

