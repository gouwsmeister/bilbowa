bilbowa: src/bilbowa.v0.1.c
	mkdir -p bin
	gcc src/bilbowa.v0.1.c -g -o bin/bilbowa -lm -pthread -Ofast -march=native -funroll-loops -w 
clean:
	rm bin/bilbowa

