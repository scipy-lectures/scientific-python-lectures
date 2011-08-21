
pyflakes: fast static analysis
===============================

They are several static analysis tools in Python; to name a few: 
`pylint <http://www.logilab.org/857>`_, 
`pychecker <http://pychecker.sourceforge.net/>`_, and 
`pyflakes <http://divmod.org/trac/wiki/DivmodPyflakes>`_.
Here we focus on pyflakes, which is the simplest tool.

    * **Fast, simple**

    * Detects syntax errors, missing imports, typos on names.

Integrating pyflakes in your editor is highly recommended, it **does
yield productivity gains**.

Running pyflakes on the current edited file
----------------------------------------------

You can bind a key to run pyflakes in the current buffer.

* **In kate**
  Menu: 'settings -> configure kate -> External Tools', add `pyflakes`:

 .. image:: pyflakes_kate.jpg
    :scale: 70

* **In vim**
  In your `.vimrc` (binds F5 to `pyflakes`)::

    autocmd FileType python let &mp = 'echo "*** running % ***" ; pyflakes %'
    autocmd FileType tex,mp,rst,python imap <Esc>[15~ <C-O>:make!^M
    autocmd FileType tex,mp,rst,python map  <Esc>[15~ :make!^M
    autocmd FileType tex,mp,rst,python set autowrite

* **In emacs**
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
        " Pyflakes"
        ;; The minor mode bindings.
        '( ([f5] . pyflakes-thisfile) )
    )
    
    (add-hook 'python-mode-hook (lambda () (pyflakes-mode t)))

A type-as-go spell-checker like integration
---------------------------------------------

* **In vim**
  Use the pyflakes.vim plugin: 
  
  1. download the zip file from
     http://www.vim.org/scripts/script.php?script_id=2441
  
  2. extract the files in `~/.vim/ftplugin/python`

  3. make sure your vimrc has "filetype plugin indent on"

* **In emacs**
  Use the flymake mode with pyflakes, documented on
  http://www.plope.com/Members/chrism/flymake-mode : add the following to
  your .emacs file::
  
    (when (load "flymake" t) 
            (defun flymake-pyflakes-init () 
            (let* ((temp-file (flymake-init-create-temp-buffer-copy 
                                'flymake-create-temp-inplace)) 
                (local-file (file-relative-name 
                            temp-file 
                            (file-name-directory buffer-file-name)))) 
                (list "pyflakes" (list local-file)))) 

            (add-to-list 'flymake-allowed-file-name-masks 
                    '("\\.py\\'" flymake-pyflakes-init))) 

    (add-hook 'find-file-hook 'flymake-find-file-hook)

