Python Scientific Lecture Notes 
===================================================================

.. raw html to center the title

.. raw:: html

  <style type="text/css">
    div.documentwrapper h1 {
        text-align: center;
        font-size: 200% ;
        font-weight: bold;
    }
  
  </style>

.. nice layout in the toc

.. raw:: html

   <SCRIPT>
   function preload_images() {
       var img = new Image();
       img.src="_images/plusBox.png";
       img.src="_images/minBox.png";
       img.src="_images/plusBoxHighlight.png";
       img.src="_images/minBoxHighlight.png";
       img.src="_images/noneBox.png";
   }
   preload_images();

   //Function to make the index toctree collapsible
   $(function () {
       $('.toctree-l3')
           .click(function(event){
               if (event.target.tagName.toLowerCase() != "a") {
                   if ($(this).children('ul').length > 0) {
                       $(this).css('list-style-image',
                       (!$(this).children('ul').is(':hidden')) ? 'url(_images/plusBoxHighlight.png)' : 'url(_images/minBoxHighlight.png)');
                       $(this).children('ul').toggle();
                   }
                   return true; //Makes links clickable
               }
           })
           .mousedown(function(event){ return false; }) //Firefox highlighting fix
           .css({cursor:'pointer', 'list-style-image':'url(_images/plusBox.png)'})
           .children('ul').hide();
       $('ul li ul li:not(:has(ul))').css({cursor:'default', 'list-style-image':'url(_images/noneBox.png)'});
       $('.toctree-l4').css({cursor:'default', 'list-style-image':'url(_images/noneBox.png)'});
       var sidebarbutton = $('#sidebarbutton');
       sidebarbutton.css({
           'display': 'none'
       });

       $('.toctree-l3').hover(
           function () {
               if ($(this).children('ul').length > 0) {
                   $(this).css('background-color', '#D0D0D0').children('ul').css('background-color', '#F0F0F0');
                   $(this).css('list-style-image',
                       (!$(this).children('ul').is(':hidden')) ? 'url(_images/minBoxHighlight.png)' : 'url(_images/plusBoxHighlight.png)');
               }
               else {
                   $(this).css('background-color', '#F9F9F9');
               }
           },
           function () {
               $(this).css('background-color', 'white').children('ul').css('background-color', 'white');
               if ($(this).children('ul').length > 0) {
                   $(this).css('list-style-image',
                       (!$(this).children('ul').is(':hidden')) ? 'url(_images/minBox.png)' : 'url(_images/plusBox.png)');
               }
           }
       );
   });

   </SCRIPT>

  <style type="text/css">
    div.bodywrapper blockquote {
        margin: 0 ;
    }

    div.toctree-wrapper ul {
        margin: 0 ;
        padding-left: 0px ;
    }

    li.toctree-l1 {
        padding: 0 ;
        list-style-type: none;
        font-size: 150% ;
        font-family: Arial, sans-serif;
        background-color: #f2f2f2;
        font-weight: normal;
        color: #20435c;
        margin-left: 0;
        margin-bottom: 1em;
        font-weight: bold;
        }

    li.toctree-l1 a {
        padding: 0 0 0 10px ;
    }

    li.toctree-l2 {
        padding: 0.25em 0 0.25em 0 ;
        list-style-type: none;
        background-color: #FFFFFF;
        font-size: 90% ;
        font-weight: bold;
    }

    li.toctree-l2 ul {
        padding-left: 40px ;
    }

    li.toctree-l3 {
        font-size: 70% ;
        list-style-type: square;
        font-weight: normal;
    }

    li.toctree-l4 {
        font-size: 85% ;
        list-style-type: circle;
        font-weight: normal;
    }

    div.topic li.toctree-l1 {
        font-size: 100% ;
        font-weight: bold;
        background-color: transparent;
        margin-bottom: 0;
        margin-left: 1.5em;
        display:inline;
    }

    div.topic p {
        font-size: 90% ;
        margin: 0.4ex;
    }

    div.topic p.topic-title {
        display:inline;
        font-size: 100% ;
        margin-bottom: 0;
    }

    div.sidebar {
        width: 25ex ;
    }

  </style>

.. only:: html

    .. sidebar:: Download 
       
       * `PDF, 2 pages per side <./_downloads/PythonScientific.pdf>`_

       * `PDF, 1 page per side <./_downloads/PythonScientific-simple.pdf>`_
   
       * `HTML and example files <https://github.com/scipy-lectures/scipy-lectures.github.com/zipball/master>`_
     
       * `Source code (github) <https://github.com/scipy-lectures/scipy-lecture-notes>`_

    .. topic:: This document

        Teaching material on the scientific Python ecosystem, a quick
        introduction to central tools and techniques. The different chapters
        each correspond to a 1 to 2 hours course with increasing level of
        expertise, from beginner to expert.

        .. toctree::
            :maxdepth: 1

            AUTHORS.rst
            CHANGES.rst
            README.rst
            LICENSE.rst

_____

.. toctree::
   :numbered:

   intro/index.rst
   advanced/index.rst

____


.. only:: html

 .. raw:: html

   <small>

 Version: |version| (output of ``git describe`` for `project repository`_)

 .. _`project repository`: https://github.com/scipy-lectures/scipy-lecture-notes

..  
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html
 
   <div style='visibility: hidden ; height=0'>

 :download:`PythonScientific.pdf` :download:`PythonScientific-simple.pdf`
 
 .. image:: themes/minBox.png
 .. image:: themes/plusBox.png
 .. image:: themes/noneBox.png
 .. image:: themes/minBoxHighlight.png
 .. image:: themes/plusBoxHighlight.png

 .. raw:: html
 
   </div>
   </small>


