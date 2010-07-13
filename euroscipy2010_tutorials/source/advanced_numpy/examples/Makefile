dist:
	rm -f advnumpy-ex.zip
	zip advnumpy-ex.zip \
		mandel.pyx mandel-answer.pyx test.wav wavreader.py \
		mandelplot.py stride-diagonals.py stride-fakedims.py \
		setup.py test.png pilbuffer.py pilbuffer-answer.py \
		myobject.c myobject_test.py setup_myobject.py \
		view-colors.py myobject-answer.c

upload: dist
	suncomet scp advnumpy-ex.zip R:public_html/tmp/advnumpy-ex.zip
