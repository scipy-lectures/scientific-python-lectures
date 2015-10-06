.. raw:: html

   <SCRIPT>
   //Function to make the index toctree collapsible
   $(function () {
       $('.toctree-l2')
           .click(function(event){
               if (event.target.tagName.toLowerCase() != "a") {
                   if ($(this).children('ul').length > 0) {
                        $(this).attr('data-content',
                            (!$(this).children('ul').is(':hidden')) ? '\u25ba' : '\u25bc');
                       $(this).children('ul').toggle();
                   }
                   return true; //Makes links clickable
               }
           })
           .mousedown(function(event){ return false; }) //Firefox highlighting fix
           .children('ul').hide();
       // Initialize the values
       $('li.toctree-l2:not(:has(ul))').attr('data-content', '-');
       $('li.toctree-l2:has(ul)').attr('data-content', '\u25ba');
       $('li.toctree-l2:has(ul)').css('cursor', 'pointer');

       $('.toctree-l2').hover(
           function () {
               if ($(this).children('ul').length > 0) {
                   $(this).css('background-color', '#D0D0D0').children('ul').css('background-color', '#F0F0F0');
                   $(this).attr('data-content',
                       (!$(this).children('ul').is(':hidden')) ? '\u25bc' : '\u25ba');
               }
               else {
                   $(this).css('background-color', '#F9F9F9');
               }
           },
           function () {
               $(this).css('background-color', 'white').children('ul').css('background-color', 'white');
               if ($(this).children('ul').length > 0) {
                   $(this).attr('data-content',
                       (!$(this).children('ul').is(':hidden')) ? '\u25bc' : '\u25ba');
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

    div.tune li, div.tune ul {
        transition-duration: 0.2s;
    }

    div.tune li.toctree-l1 {
        padding: 5px 0 0;
        list-style-type: none;
        font-size: 150% ;
        font-family: Arial, sans-serif;
        background-color: #f2f2f2;
        font-weight: normal;
        color: #20435c;
        margin-left: 0;
        margin-bottom: 1.2em;
        font-weight: bold;
        }

    div.tune li.toctree-l1 a {
        padding: 0 0 0 10px ;
        color: #314F64 ;
    }

    div.tune li.toctree-l2 {
        padding: 0.25em 0 0.25em 0 ;
        list-style-type: none;
        background-color: #FFFFFF;
        font-size: 85% ;
        font-weight: normal;
    }

    div.tune li.toctree-l2 ul {
        padding-left: 40px ;
    }


    div.tune li.toctree-l2:before {
        content: attr(data-content) ;
        font-size: 85% ;
        color: #777 ;
        display: inline-block;
        width: 10px;
    }

    div.tune li.toctree-l3 {
        font-size: 75% ;
        list-style-type: square;
        font-weight: normal;
    }

    div.tune li.toctree-l4 {
        font-size: 85% ;
        list-style-type: circle;
        font-weight: normal;
    }

    div.preface li.toctree-l1 {
        font-weight: bold;
        background-color: transparent;
        margin-bottom: 0;
        margin-left: 0;
        margin-top: 10px;
        display: inline;
    }

    div.preface li.toctree-l2 {
        background-color: transparent;
        margin-bottom: 0;
        margin-left: 1.5em;
        display: inline;
    }

    div.sidebar {
        width: 25ex ;
    }

    @media 	only screen and (max-width: 1080px) and (-webkit-min-device-pixel-ratio: 2.5), (max-width: 70ex) {
        div.sidebar {
            width: 79%;
            padding-left: 15%;
            font-size: 70%;
        }
    }

  @media (max-width: 1324px) {
        div.sphinxsidebar {
            display: none;
        }
  }

  </style>


