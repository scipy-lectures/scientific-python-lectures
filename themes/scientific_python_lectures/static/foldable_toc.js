// Function to make the index toctree foldable
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
