#!/bin/sh

xrdb -merge - <<OEXRDB
XTerm*VT100.translations: #override \
    <Shift>Up:	scroll-back(1,line)\n\
    <Shift>Down:    scroll-forw(1,line)\n\
    <Key>BackSpace: string(0x7f) \n\
    <Key>Delete:    string(0x1b) string([3~) \n\
    <Key>Home:	string(0x1b) string([1~) \n\
    <Key>End:	string(0x1b) string([4~) \n\
    <Key>KP_Home:   string(0x1b) string([1~) \n\
    <Key>KP_End:    string(0x1b) string([4~) \n\
    <Key>KP_Delete: string(0x1b) string([3~) \n\
    <Key>KP_Enter:  string(0x0d) \n\
    <Ctrl>-:      smaller-vt-font() \n\
    <Ctrl>+:  larger-vt-font() \n\
    <Key>Shift_L: select-cursor-start() \ select-cursor-end(CLIPBOARD,CUT_BUFFER0,PRIMARY)\n\

*.VT100.VeryBoldColors:	6

XTerm*Scrollbar*width:		7
XTerm*Scrollbar*height:		7
XTerm*Scrollbar*shadowWidth:	2
XTerm*Scrollbar*borderWidth:	3
XTerm*ScrollBar: on
XTerm*SaveLines: 2000
XTerm*termName: xterm-color
XTerm*scrollKey:	on
XTerm*scrollTtyOutput:	off

XTerm*cursorColor:      firebrick3
XTerm*foreground:	Black
XTerm*background:	white

XTerm*color0:	black
XTerm*color1:	red3
XTerm*color2:	rgb:0/60/0
XTerm*color3:	yellow4
XTerm*color4:	blue4
XTerm*color5:	magenta3
XTerm*color6:	DeepSkyBlue4
XTerm*color7:	gray95
XTerm*color8:	gray30
XTerm*color9:	red
XTerm*color10:	rgb:10/70/10
XTerm*color11:	yellow2
XTerm*color12:	blue3
XTerm*color13:	magenta
XTerm*color14:	DeepSkyblue3
XTerm*color15:	gray90
XTerm*.font:	-*-fixed-medium-r-*-*-24-*-*-*-*-*-*-*
XTerm*.boldfont:	-*-fixed-bold-r-*-*-24-*-*-*-*-*-*-*


yeahconsole*consoleHeight: 16
yeahconsole*aniDelay: 0
OEXRDB

/usr/bin/yeahconsole
