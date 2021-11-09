#
# Copyright (c) 2018-2021 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to use this Software with Cadence processor cores only and
# not with any other processors and platforms, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

.PHONY: all install clean
include detect_core.mk

all: nnlib annlib

install: install_nnlib install_annlib

clean: clean_nnlib clean_annlib


nnlib:
	$(MAKE) -f makefile_nn_lib_hifi4 all

install_nnlib: nnlib
	$(MAKE) -f makefile_nn_lib_hifi4 install

clean_nnlib:
	$(MAKE) -f makefile_nn_lib_hifi4 clean

ifeq ($(xclib), 1)
annlib:
	$(MAKE) -f makefile_ann_lib_hifi4 all 

install_annlib: annlib
	$(MAKE) -f makefile_ann_lib_hifi4 install

clean_annlib:
	$(MAKE) -f makefile_ann_lib_hifi4 clean 

else
annlib install_annlib clean_annlib:
	@echo "ann_lib is not supported with newlib"
endif
