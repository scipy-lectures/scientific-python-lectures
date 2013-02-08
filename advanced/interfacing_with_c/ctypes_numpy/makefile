m.PHONY : clean

libcos_doubles.so : cos_doubles.o
	gcc -shared -Wl,-soname,libcos_doubles.so -o libcos_doubles.so cos_doubles.o

cos_doubles.o : cos_doubles.c
	gcc -c -fPIC cos_doubles.c -o cos_doubles.o

clean :
	-rm -vf libcos_doubles.so cos_doubles.o cos_doubles.pyc
