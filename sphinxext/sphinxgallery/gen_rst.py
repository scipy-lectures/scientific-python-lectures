# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Generate the rst files for the examples by iterating over the python
example files.

Files that generate images should start with 'plot'

"""
from __future__ import division, print_function, absolute_import
from time import time
import os
import re
import shutil
import traceback
import glob
import sys
import subprocess
import warnings
from . import path_static as glr_path_static
from .backreferences import write_backreferences, _thumbnail_div


# Try Python 2 first, otherwise load from Python 3
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


try:
    # Python 2 built-in
    execfile
except NameError:
    def execfile(filename, global_vars=None, local_vars=None):
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, global_vars, local_vars)

try:
    basestring
except NameError:
    basestring = str

import token
import tokenize
import numpy as np

try:
    # make sure that the Agg backend is set before importing any
    # matplotlib
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    # this script can be imported by nosetest to find tests to run: we should
    # not impose the matplotlib requirement in that case.
    pass


###############################################################################

class Tee(object):
    """A tee object to redirect streams to multiple outputs"""
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


###############################################################################
rst_template = """

.. _example_%(short_fname)s:

%(docstring)s

**Python source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
    :lines: %(end_row)s-
    """

plot_rst_template = """

.. _example_%(short_fname)s:

%(docstring)s

%(image_list)s

%(stdout)s

**Python source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
    :lines: %(end_row)s-

**Total running time of the example:** %(time_elapsed) .2f seconds
(%(time_m) .0f minutes %(time_s) .2f seconds)
    """

# The following strings are used when we have several pictures: we use
# an html div tag that our CSS uses to turn the lists into horizontal
# lists.
HLIST_HEADER = """
.. rst-class:: sphx-glr-horizontal

"""

HLIST_IMAGE_TEMPLATE = """
    *

      .. image:: images/%s
            :scale: 47
"""

SINGLE_IMAGE = """
.. image:: images/%s
    :align: center
"""


def extract_docstring(filename, ignore_heading=False):
    """ Extract a module-level docstring, if any
    """
    lines = open(filename).readlines()
    start_row = 0
    if lines[0].startswith('#!'):
        lines.pop(0)
        start_row = 1
    docstring = ''
    first_par = ''
    line_iterator = iter(lines)
    tokens = tokenize.generate_tokens(lambda: next(line_iterator))
    for tok_type, tok_content, _, (erow, _), _ in tokens:
        tok_type = token.tok_name[tok_type]
        if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
            continue
        elif tok_type == 'STRING':
            docstring = eval(tok_content)
            # If the docstring is formatted with several paragraphs, extract
            # the first one:
            paragraphs = '\n'.join(
                line.rstrip() for line
                in docstring.split('\n')).split('\n\n')
            if paragraphs:
                if ignore_heading:
                    if len(paragraphs) > 1:
                        first_par = re.sub('\n', ' ', paragraphs[1])
                        first_par = ((first_par[:95] + '...')
                                     if len(first_par) > 95 else first_par)
                    else:
                        raise ValueError("Docstring not found by gallery.\n"
                                         "Please check the layout of your"
                                         " example file:\n {}\n and make sure"
                                         " it's correct".format(filename))
                else:
                    first_par = paragraphs[0]

        break
    return docstring, first_par, erow + 1 + start_row


def extract_line_count(filename, target_dir):
    """Extract the line count of a file"""
    example_file = os.path.join(target_dir, filename)
    lines = open(example_file).readlines()
    start_row = 0
    if lines and lines[0].startswith('#!'):
        lines.pop(0)
        start_row = 1
    line_iterator = iter(lines)
    tokens = tokenize.generate_tokens(lambda: next(line_iterator))
    check_docstring = True
    erow_docstring = 0
    for tok_type, _, _, (erow, _), _ in tokens:
        tok_type = token.tok_name[tok_type]
        if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
            continue
        elif (tok_type == 'STRING') and check_docstring:
            erow_docstring = erow
            check_docstring = False
    return erow_docstring+1+start_row, erow+1+start_row


def line_count_sort(file_list, target_dir):
    """Sort the list of examples by line-count"""
    new_list = [x for x in file_list if x.endswith('.py')]
    unsorted = np.zeros(shape=(len(new_list), 2))
    unsorted = unsorted.astype(np.object)
    for count, exmpl in enumerate(new_list):
        docstr_lines, total_lines = extract_line_count(exmpl, target_dir)
        unsorted[count][1] = total_lines - docstr_lines
        unsorted[count][0] = exmpl
    index = np.lexsort((unsorted[:, 0].astype(np.str),
                        unsorted[:, 1].astype(np.float)))
    if not len(unsorted):
        return []
    return np.array(unsorted[index][:, 0]).tolist()


def generate_dir_rst(src_dir, target_dir, gallery_conf,
                     plot_gallery, seen_backrefs):
    """Generate the rst file for an example directory"""
    if not os.path.exists(os.path.join(src_dir, 'README.txt')):
        print(80 * '_')
        print('Example directory %s does not have a README.txt file' %
              src_dir)
        print('Skipping this directory')
        print(80 * '_')
        return ""  # because string is an expected return type

    fhindex = open(os.path.join(src_dir, 'README.txt')).read()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    sorted_listdir = line_count_sort(os.listdir(src_dir),
                                     src_dir)
    for fname in sorted_listdir:
        if fname.endswith('py'):
            generate_file_rst(fname, target_dir, src_dir, plot_gallery)
            new_fname = os.path.join(src_dir, fname)
            _, snippet, _ = extract_docstring(new_fname, True)
            write_backreferences(seen_backrefs, gallery_conf,
                               target_dir, fname, snippet)

            fhindex += _thumbnail_div(target_dir, fname, snippet)
            fhindex += """

.. toctree::
   :hidden:

   /%s/%s\n""" % (target_dir, fname[:-3])


# clear at the end of the section
    fhindex += """.. raw:: html\n
    <div style='clear:both'></div>\n\n"""

    return fhindex


def scale_image(in_fname, out_fname, max_width, max_height):
    """Scales an image with the same aspect ratio centered in an
       image with a given max_width and max_height
       if in_fname == out_fname the image can only be scaled down
    """
    # local import to avoid testing dependency on PIL:
    try:
        from PIL import Image
    except ImportError:
        import Image
    img = Image.open(in_fname)
    width_in, height_in = img.size
    scale_w = max_width / float(width_in)
    scale_h = max_height / float(height_in)

    if height_in * scale_w <= max_height:
        scale = scale_w
    else:
        scale = scale_h

    if scale >= 1.0 and in_fname == out_fname:
        return

    width_sc = int(round(scale * width_in))
    height_sc = int(round(scale * height_in))

    # resize the image
    img.thumbnail((width_sc, height_sc), Image.ANTIALIAS)

    # insert centered
    thumb = Image.new('RGB', (max_width, max_height), (255, 255, 255))
    pos_insert = ((max_width - width_sc) // 2, (max_height - height_sc) // 2)
    thumb.paste(img, pos_insert)

    thumb.save(out_fname)
    # Use optipng to perform lossless compression on the resized image if
    # software is installed
    if os.environ.get('SKLEARN_DOC_OPTIPNG', False):
        try:
            subprocess.call(["optipng", "-quiet", "-o", "9", out_fname])
        except Exception:
            warnings.warn('Install optipng to reduce the size of the \
                          generated images')


def execute_script(image_dir, thumb_file, image_fname, base_image_name,
                   src_file, fname):
    image_path = os.path.join(image_dir, image_fname)
    stdout_path = os.path.join(image_dir,
                               'stdout_%s.txt' % base_image_name)
    time_path = os.path.join(image_dir,
                             'time_%s.txt' % base_image_name)
    # The following is a list containing all the figure names
    time_elapsed = 0
    figure_list = []
    first_image_file = image_path % 1
    if os.path.exists(stdout_path):
        stdout = open(stdout_path).read()
    else:
        stdout = ''
    if os.path.exists(time_path):
        time_elapsed = float(open(time_path).read())

    if not os.path.exists(first_image_file) or \
       os.stat(first_image_file).st_mtime <= os.stat(src_file).st_mtime:
        # We need to execute the code
        print('plotting %s' % fname)
        t0 = time()
        import matplotlib.pyplot as plt
        plt.close('all')
        cwd = os.getcwd()
        try:
            # First CD in the original example dir, so that any file
            # created by the example get created in this directory
            orig_stdout = sys.stdout
            os.chdir(os.path.dirname(src_file))
            my_buffer = StringIO()
            my_stdout = Tee(sys.stdout, my_buffer)
            sys.stdout = my_stdout
            my_globals = {'pl': plt, '__name__': 'gallery'}
            execfile(os.path.basename(src_file), my_globals)
            time_elapsed = time() - t0
            sys.stdout = orig_stdout
            my_stdout = my_buffer.getvalue()

            if '__doc__' in my_globals:
                # The __doc__ is often printed in the example, we
                # don't with to echo it
                my_stdout = my_stdout.replace(
                    my_globals['__doc__'],
                    '')
            my_stdout = my_stdout.strip().expandtabs()
            if my_stdout:
                stdout = """**Script output**:\n
.. rst-class:: sphx-glr-script-out

  ::

    {}\n""".format('\n    '.join(my_stdout.split('\n')))
            os.chdir(cwd)
            open(stdout_path, 'w').write(stdout)
            open(time_path, 'w').write('%f' % time_elapsed)

            # In order to save every figure we have two solutions :
            # * iterate from 1 to infinity and call plt.fignum_exists(n)
            #   (this requires the figures to be numbered
            #    incrementally: 1, 2, 3 and not 1, 2, 5)
            # * iterate over [fig_mngr.num for fig_mngr in
            #   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
            fig_managers = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
            for fig_mngr in fig_managers:
                # Set the fig_num figure as the current figure as we can't
                # save a figure that's not the current figure.
                fig = plt.figure(fig_mngr.num)
                kwargs = {}
                to_rgba = matplotlib.colors.colorConverter.to_rgba
                for attr in ['facecolor', 'edgecolor']:
                    fig_attr = getattr(fig, 'get_' + attr)()
                    default_attr = matplotlib.rcParams['figure.' + attr]
                    if to_rgba(fig_attr) != to_rgba(default_attr):
                        kwargs[attr] = fig_attr

                fig.savefig(image_path % fig_mngr.num, **kwargs)
                figure_list.append(image_fname % fig_mngr.num)
        except:
            print(80 * '_')
            print('%s is not compiling:' % fname)
            traceback.print_exc()
            print(80 * '_')
        finally:
            os.chdir(cwd)
            sys.stdout = orig_stdout

        print(" - time elapsed : %.2g sec" % time_elapsed)
    else:
        figure_list = [f[len(image_dir):]
                       for f in glob.glob(image_path.replace("%03d",
                                            '[0-9][0-9][0-9]'))]
    figure_list.sort()

    # generate thumb file
    if os.path.exists(first_image_file):
        scale_image(first_image_file, thumb_file, 400, 280)

    # Depending on whether we have one or more figures, we're using a
    # horizontal list or a single rst call to 'image'.
    if len(figure_list) == 1:
        figure_name = figure_list[0]
        image_list = SINGLE_IMAGE % figure_name.lstrip('/')
    else:
        image_list = HLIST_HEADER
        for figure_name in figure_list:
            image_list += HLIST_IMAGE_TEMPLATE % figure_name.lstrip('/')

    return image_list, time_elapsed, stdout


def generate_file_rst(fname, target_dir, src_dir, plot_gallery):
    """ Generate the rst file for a given example."""
    base_image_name = os.path.splitext(fname)[0]
    image_fname = 'sphx_glr_%s_%%03d.png' % base_image_name

    this_template = rst_template
    short_fname = target_dir.replace(os.path.sep, '_') + '_' + fname
    src_file = os.path.join(src_dir, fname)
    example_file = os.path.join(target_dir, fname)
    shutil.copyfile(src_file, example_file)

    image_dir = os.path.join(target_dir, 'images')
    thumb_dir = os.path.join(image_dir, 'thumb')
    thumb_file = os.path.join(thumb_dir, 'sphx_glr_%s_thumb.png' % base_image_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)

    time_elapsed = 0
    if plot_gallery and fname.startswith('plot'):
        # generate the plot as png image if file name
        # starts with plot and if it is more recent than an
        # existing image.
        image_list, time_elapsed, stdout = execute_script(image_dir,
                                                          thumb_file,
                                                          image_fname,
                                                          base_image_name,
                                                          src_file, fname)
        this_template = plot_rst_template

    if not os.path.exists(thumb_file):
        # create something to replace the thumbnail
        scale_image(os.path.join(glr_path_static(), 'no_image.png'),
                    thumb_file, 200, 140)

    docstring, short_desc, end_row = extract_docstring(example_file)

    time_m, time_s = divmod(time_elapsed, 60)
    f = open(os.path.join(target_dir, base_image_name + '.rst'), 'w')
    f.write(this_template % locals())
    f.flush()
