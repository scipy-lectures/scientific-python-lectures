/*  Example of wrapping the cos function from math.h using the NumPy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


/*  wrapped cosine function */
static PyObject* cos_func_np(PyObject* self, PyObject* args)
{
    PyArrayObject *arrays[2];  /* holds input and output array */
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[2];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[2];

    NpyIter_IterNextFunc *iternext;

    /*  parse single NumPy array argument */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrays[0])) {
        return NULL;
    }

    arrays[1] = NULL;  /* The result will be allocated by the iterator */

    /* Set up and create the iterator */
    iterator_flags = (NPY_ITER_ZEROSIZE_OK |
                      /*
                       * Enable buffering in case the input is not behaved
                       * (native byte order or not aligned),
                       * disabling may speed up some cases when it is known to
                       * be unnecessary.
                       */
                      NPY_ITER_BUFFERED |
                      /* Manually handle innermost iteration for speed: */
                      NPY_ITER_EXTERNAL_LOOP |
                      NPY_ITER_GROWINNER);

    op_flags[0] = (NPY_ITER_READONLY |
                   /*
                    * Required that the arrays are well behaved, since the cos
                    * call below requires this.
                    */
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);

    /* Ask the iterator to allocate an array to write the output to */
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    /*
     * Ensure the iteration has the correct type, could be checked
     * specifically here.
     */
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];

    /* Create the NumPy iterator object: */
    iter = NpyIter_MultiNew(2, arrays, iterator_flags,
                            /* Use input order for output and iteration */
                            NPY_KEEPORDER,
                            /* Allow only byte-swapping of input */
                            NPY_EQUIV_CASTING, op_flags, op_dtypes);
    Py_DECREF(op_dtypes[0]);  /* The second one is identical. */

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    /* Fetch the output array which was allocated by the iterator: */
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

    if (NpyIter_GetIterSize(iter) == 0) {
        /*
         * If there are no elements, the loop cannot be iterated.
         * This check is necessary with NPY_ITER_ZEROSIZE_OK.
         */
        NpyIter_Deallocate(iter);
        return ret;
    }

    /* The location of the data pointer which the iterator may update */
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* iterate over the arrays */
    do {
        npy_intp stride = strideptr[0];
        npy_intp count = *innersizeptr;
        /* out is always contiguous, so use double */
        double *out = (double *)dataptr[1];
        char *in = dataptr[0];

        /* The output is allocated and guaranteed contiguous (out++ works): */
        assert(strideptr[1] == sizeof(double));

        /*
         * For optimization it can make sense to add a check for
         * stride == sizeof(double) to allow the compiler to optimize for that.
         */
        while (count--) {
            *out = cos(*(double *)in);
            out++;
            in += stride;
        }
    } while (iternext(iter));

    /* Clean up and return the result */
    NpyIter_Deallocate(iter);
    return ret;
}


/*  define functions in module */
static PyMethodDef CosMethods[] =
{
     {"cos_func_np", cos_func_np, METH_VARARGS,
         "evaluate the cosine on a NumPy array"},
     {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "cos_module", "Some documentation",
    -1,
    CosMethods
};
PyMODINIT_FUNC PyInit_cos_module_np(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module==NULL) return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}

#else
/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initcos_module_np(void) {
    PyObject *module;
    module = Py_InitModule("cos_module_np", CosMethods);
    if(module==NULL) return;
    /* IMPORTANT: this must be called */
    import_array();
    return;
}

#endif
