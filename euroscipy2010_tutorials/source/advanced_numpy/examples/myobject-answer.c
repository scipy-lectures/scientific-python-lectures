/*
 * Sample implementation of a custom data type that exposes an array
 * interface. (And does nothing else :)
 *
 * Requires Python >= 3.1
 */

/*
 * Mini-exercises:
 *
 * - make the array strided
 * - change the data type
 *
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


typedef struct {
    PyObject_HEAD
    int buffer[4];
} PyMyObjectObject;


static int
myobject_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PyMyObjectObject *self = (PyMyObjectObject*)obj;

    /* Called when something requests that a MyObject-type object 
       provides a buffer interface */

    view->buf = self->buffer;
    view->readonly = 0;
    view->format = "i";
    view->len = 4;
    view->itemsize = sizeof(int);
    view->ndim = 2;
    view->shape = malloc(sizeof(Py_ssize_t) * 2);
    view->shape[0] = 2;
    view->shape[1] = 2;
    view->strides = malloc(sizeof(Py_ssize_t) * 2);;
    view->strides[0] = 2*sizeof(int);
    view->strides[1] = sizeof(int);
    view->suboffsets = NULL;

    /* Note: if correct interpretation *requires* strides or shape,
       you need to check flags for what was requested, and raise 
       appropriate errors. 
          
       The same if the buffer is not readable. 
    */ 

    view->obj = (PyObject*)self;
    Py_INCREF(self);

    return 0;
}

static void
myobject_releasebuffer(PyMemoryViewObject *self, Py_buffer *view)
{
    if (view->shape) {
        free(view->shape); 
        view->shape = NULL;
    }
    if (view->strides) {
        free(view->strides); 
        view->strides = NULL;
    }
}


static PyBufferProcs myobject_as_buffer = {
    (getbufferproc)myobject_getbuffer,
    (releasebufferproc)myobject_releasebuffer,
};

/*
 * Standard stuff follows
 */

PyTypeObject PyMyObject_Type;

static PyObject *
myobject_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PyMyObjectObject *self;
    static char *kwlist[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) {
        return NULL;
    }
    self = (PyMyObjectObject *)
        PyObject_New(PyMyObjectObject, &PyMyObject_Type);
    self->buffer[0] = 1;
    self->buffer[1] = 2;
    self->buffer[2] = 3;
    self->buffer[3] = 4;
    return (PyObject*)self;
}


PyTypeObject PyMyObject_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MyObject",
    sizeof(PyMyObjectObject),
    0,                                          /* tp_itemsize */
    /* methods */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    &myobject_as_buffer,                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    myobject_new,                               /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};

struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "myobject",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit_myobject(void) {
    PyObject *m, *d;

    if (PyType_Ready(&PyMyObject_Type) < 0) {
        return NULL;
    }
    
    m = PyModule_Create(&moduledef);

    d = PyModule_GetDict(m);
    
    Py_INCREF(&PyMyObject_Type);
    PyDict_SetItemString(d, "MyObject", (PyObject *)&PyMyObject_Type);
    
    return m;
}
