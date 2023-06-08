// Highlight the table of content as we scroll

$( document ).ready(function () {
    sections = {},
    i        = 0,
    url = document.URL.replace(/#.*$/, ""),
    current_section = 0;

    // Grab positions of our sections
    $('.headerlink').each(function(){
        sections[this.href.replace(url, '')] = $(this).offset().top - 50;
    });

    $(window).scroll(function(event) {
    var pos   = $(window).scrollTop();

    // Highlight the current section
    $('a.internal').parent().removeClass('active');
        for(i in sections){
            if(sections[i] > pos){
            break;
            };
        if($('a.internal[href$="' + i + '"]').is(':visible')){
            current_section = i;
        };
        }
    $('a.internal[href$="' + current_section + '"]').parent().addClass('active');
    $('a.internal[href$="' + current_section + '"]').parent().parent().parent().addClass('active');
    $('a.internal[href$="' + current_section + '"]').parent().parent().parent().parent().parent().addClass('active');
    });
});
