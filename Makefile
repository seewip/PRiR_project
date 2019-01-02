CXX = nvcc
LIBFLAG = `pkg-config opencv --cflags --libs`

main: main.cu
	$(CXX) main.cu -o main $(LIBFLAG)

clean:
	rm main
