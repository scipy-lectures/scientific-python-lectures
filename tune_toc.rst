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


