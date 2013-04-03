/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/*  wrapped cosine function */
static PyObject* cos_func_np(PyObject* self, PyObject* args)
{

    PyArrayObject *in_array;
    PyObject      *out_array;
    PyArrayIterObject *in_iter;
    PyArrayIterObject *out_iter;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array))
        return NULL;

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    /* TODO: this iterator API is deprecated since 1.6
     *       replace in favour of the new NpyIter API */
    in_iter  = (PyArrayIterObject *)PyArray_IterNew((PyObject*)in_array);
    out_iter = (PyArrayIterObject *)PyArray_IterNew(out_array);
    if (in_iter == NULL || out_iter == NULL)
        goto fail;

    /*  iterate over the arrays */
    while (in_iter->index < in_iter->size
            && out_iter->index < out_iter->size) {
        /* get the datapointers */
        double * in_dataptr = (double *)in_iter->dataptr;
        double * out_dataptr = (double *)out_iter->dataptr;
        /* cosine of input into output */
        *out_dataptr = cos(*in_dataptr);
        /* update the iterator */
        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    /*  clean up and return the result */
    Py_DECREF(in_iter);
    Py_DECREF(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        Py_XDECREF(in_iter);
        Py_XDECREF(out_iter);
        return NULL;
}

/*  define functions in module */
static PyMethodDef CosMethods[] =
{
     {"cos_func_np", cos_func_np, METH_VARARGS,
         "evaluate the cosine on a numpy array"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initcos_module_np(void)
{
     (void) Py_InitModule("cos_module_np", CosMethods);
     /* IMPORTANT: this must be called */
     import_array();
}
