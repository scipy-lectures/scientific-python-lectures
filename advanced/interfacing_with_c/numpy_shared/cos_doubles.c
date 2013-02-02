#include <math.h>

void cos_doubles(double * in, double * out, int size){
    int i;
    for(i=0;i<size;i++){
        out[i] = cos(in[i]);
    }
}
