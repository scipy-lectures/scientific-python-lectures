# -*- ksh -*-
#
# If you use the GNU debugger gdb to debug the Python C runtime, you
# might find some of the following commands useful.  Copy this to your
# ~/.gdbinit file and it'll get loaded into gdb automatically when you
# start it up.  Then, at the gdb prompt you can do things like:
#
#    (gdb) pyo apyobjectptr
#    <module 'foobar' (built-in)>
#    refcounts: 1
#    address    : 84a7a2c
#    $1 = void
#    (gdb)

# Prints a representation of the object to stderr, along with the
# number of reference counts it current has and the hex address the
# object is allocated at.  The argument must be a PyObject*
define pyo
print _PyObject_Dump($arg0)
end

# Prints a representation of the object to stderr, along with the
# number of reference counts it current has and the hex address the
# object is allocated at.  The argument must be a PyGC_Head*
define pyg
print _PyGC_Dump($arg0)
end

# A rewrite of the Python interpreter's line number calculator in GDB's
# command language
define lineno
    set $__continue = 1
    set $__co = f->f_code
    set $__lasti = f->f_lasti
    set $__sz = ((PyStringObject *)$__co->co_lnotab)->ob_size/2
    set $__p = (unsigned char *)((PyStringObject *)$__co->co_lnotab)->ob_sval
    set $__li = $__co->co_firstlineno
    set $__ad = 0
    while ($__sz-1 >= 0 && $__continue)
      set $__sz = $__sz - 1
      set $__ad = $__ad + *$__p
      set $__p = $__p + 1
      if ($__ad > $__lasti)
	set $__continue = 0
      end
      set $__li = $__li + *$__p
      set $__p = $__p + 1
    end
    printf "%d", $__li
end

define pyframe
    set $__fn = (char *)((PyStringObject *)co->co_filename)->ob_sval
    set $__n = (char *)((PyStringObject *)co->co_name)->ob_sval
    printf "%s (", $__fn
    lineno
    printf "): %s\n", $__n
### Uncomment these lines when using from within Emacs/XEmacs so it will
### automatically track/display the current Python source line
#    printf "%c%c%s:", 032, 032, $__fn
#    lineno
#    printf ":1\n"
end

define printframe
    if $pc > PyEval_EvalFrameEx && $pc < PyEval_EvalCodeEx
	pyframe
    else
        frame
    end
end

