#!/bin/sh
# Script to do a local install of sphinx-gallery
rm -rf tmp sphinxgallery
mkdir -p tmp/lib/python2.7/site-packages

easy_install -Zeab tmp sphinx-gallery

cp -vr tmp/sphinx-gallery/sphinxgallery .

git add sphinxgallery 