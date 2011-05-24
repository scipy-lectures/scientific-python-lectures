
pyflakes: fast static analysis
===============================

* **Fast, simple**

* Detects syntax errors, missing imports, typos on names.

In kate
--------

Menu: 'settings -> configure kate -> External Tools', add `pyflakes`:

.. image:: pyflakes_kate.jpg

In vim
--------

In your `.vimrc` (binds F5 to `pyflakes`)::

    autocmd FileType python let &mp = 'echo "*** running % ***" ; pyflakes %'
    autocmd FileType tex,mp,rst,python imap <Esc>[15~ <C-O>:make!^M
    autocmd FileType tex,mp,rst,python map  <Esc>[15~ :make!^M
    autocmd FileType tex,mp,rst,python set autowrite

In emacs
---------

In your `.emacs` (binds F5 to `pyflakes`)::

    (defun pyflakes-thisfile () (interactive)
           (compile (format "pyflakes %s" (buffer-file-name)))
    )
    
    (define-minor-mode pyflakes-mode
        "Toggle pyflakes mode.
        With no argument, this command toggles the mode.
        Non-null prefix argument turns on the mode.
        Null prefix argument turns off the mode."
        ;; The initial value.
        nil
        ;; The indicator for the mode line.
        " Pyfalkes"
        ;; The minor mode bindings.
        '( ([f5] . pyflakes-thisfile) )
    )
    
    (add-hook 'python-mode-hook (lambda () (pyflakes-mode t)))

