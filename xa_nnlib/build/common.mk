#
# Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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

QUIET =
WARNING_AS_ERROR ?= 1
MAPFILE  = map_$(CODEC_NAME).txt
LDSCRIPT = ldscript_$(CODEC_NAME).txt
SYMFILE  = symbols_$(CODEC_NAME).txt
DETECTED_CORE?=

ifeq ($(CPU), x86)
    S = /
    AR = ar
    OBJCOPY = objcopy
    CC = gcc
    CXX = g++
    CFLAGS += -fno-exceptions -DCSTUB=1
    CFLAGS += -ffloat-store 
    CFLAGS += -DHIFI3_CSTUB
    RM = rm -f
    RM_R = rm -rf
    MKPATH = mkdir -p
    CP = cp -f
    INCLUDES += \
    -I$(ROOTDIR)/test/include
	IFEXIST =
else
    #switch to xt-clang default RI.7
    AR = xt-ar $(XTCORE)
    OBJCOPY = xt-objcopy $(XTCORE)
    #CC = xt-xcc $(XTCORE)
    #CXX = xt-xc++ $(XTCORE)
    CC = xt-clang $(XTCORE)
    CXX = xt-clang++ $(XTCORE)
    ISS = xt-run $(XTCORE)
    CONFIGDIR := $(shell $(ISS) --show-config=config)
    include $(CONFIGDIR)/misc/hostenv.mk
	GREPARGS =
	WINNUL =
	IFEXIST =
	ifeq ($(HOSTTYPE),win)
	GREPARGS = /c:
	WINNUL = 2>nul
	IFEXIST = if exist
	endif
	has_mul16_tmp = $(shell $(GREP) $(GREPARGS)"IsaUseMul16 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
	has_mul32_tmp = $(shell $(GREP) $(GREPARGS)"IsaUse32bitMul = 1" "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
	has_mul16=1
	has_mul32=1
	ifeq (,$(has_mul16_tmp))
	has_mul16=0
	endif
	ifeq (,$(has_mul32_tmp))
	has_mul32=0
	endif
    CFLAGS += -Wall 
    ifeq ($(WARNING_AS_ERROR),1)
      CFLAGS += -Werror
      ifneq ($(CC), xt-xcc)
       CFLAGS += -Wno-parentheses-equality
      endif
    endif
    ifeq "$(has_mul16)" "0"
        CFLAGS += -mno-mul16
    endif
    ifeq "$(has_mul32)" "0"
        CFLAGS += -mno-mul32 -mno-div32
    endif
    CFLAGS += -fsigned-char -fno-exceptions -mlongcalls -INLINE:requested -mcoproc -fno-zero-initialized-in-bss
    CFLAGS += -mtext-section-literals 
    CFLAGS += -Wsign-compare
endif

OBJDIR = objs$(S)$(CODEC_NAME)$(DETECTED_CORE)
LIBDIR = $(ROOTDIR)$(S)lib

OBJ_LIBO2OBJS = $(addprefix $(OBJDIR)/,$(LIBO2OBJS))
OBJ_LIBOSOBJS = $(addprefix $(OBJDIR)/,$(LIBOSOBJS))
OBJ_LIBO2CPPOBJS = $(addprefix $(OBJDIR)/,$(LIBO2CPPOBJS))
OBJ_LIBO2CCOBJS = $(addprefix $(OBJDIR)/,$(LIBO2CCOBJS))
OBJ_LIBOSCPPOBJS = $(addprefix $(OBJDIR)/,$(LIBOSCPPOBJS))

#$(info OBJ_LIBO2OBJS="$(OBJ_LIBO2OBJS)")

ALL_OBJS := \
  $(OBJ_LIBO2OBJS) \
  $(OBJ_LIBOSOBJS) \
  $(OBJ_LIBO2CPPOBJS) \
  $(OBJ_LIBOSCPPOBJS) \
  $(OBJ_LIBO2CCOBJS) \

ALL_DEPS := $(foreach dep,$(ALL_OBJS),${dep:%.o=%.d})
-include $(ALL_DEPS)

TEMPOBJ = temp.o    

ifeq ($(CPU), x86)
    LIBOBJ   = $(OBJDIR)/xgcc_$(CODEC_NAME)$(DETECTED_CORE).o
    LIB      = xgcc_$(CODEC_NAME)$(DETECTED_CORE).a
else
    LIBOBJ   = $(OBJDIR)/xa_$(CODEC_NAME)$(DETECTED_CORE).o
    LIB      = xa_$(CODEC_NAME)$(DETECTED_CORE).a
endif


CFLAGS += $(EXTRA_CFLAGS) $(EXTRA_CFLAGS2)

LIBLDFLAGS += \
    $(EXTRA_LIBLDFLAGS)

ifeq ($(DEBUG),1)
  NOSTRIP = 1
  OPT_O2 = -O0 -g 
  OPT_OS = -O0 -g
  OPT_O0 = -O0 -g 
  CFLAGS += -DDEBUG
else
ifeq ($(CPU), x86)
  OPT_O2 = -O2 -g 
  OPT_OS = -O2 -g 
  OPT_O0 = -O0 -g 
else
  OPT_O2 = -O3 -LNO:simd 
  OPT_OS = -Os 
  OPT_O0 = -O0 
  CFLAGS += -DNDEBUG=1
endif
endif


all: $(OBJDIR) $(LIB)

install: $(LIB)
	@echo "Installing $(LIB)"
	$(QUIET) -$(MKPATH) "$(LIBDIR)"
	$(QUIET) $(CP) $(LIB) "$(LIBDIR)"

$(OBJDIR):
	$(QUIET) -$(MKPATH) $@

ifeq ($(NOSTRIP), 1)
$(LIBOBJ): $(OBJ_LIBO2OBJS) $(OBJ_LIBOSOBJS) $(OBJ_LIBO2CPPOBJS) $(OBJ_LIBO2CCOBJS) $(OBJ_LIBOSCPPOBJS) 
	@echo "Linking Objects"
	$(QUIET) $(CC) -o $@ $^ \
	-Wl,-r,-Map,$(MAPFILE) --no-standard-libraries
else
$(LIBOBJ): $(OBJ_LIBO2OBJS) $(OBJ_LIBOSOBJS) $(OBJ_LIBO2CPPOBJS) $(OBJ_LIBO2CCOBJS) $(OBJ_LIBOSCPPOBJS)
	@echo "Linking Objects"
	$(QUIET) $(CC) -o $@ $^ \
	-Wl,-r,-Map,$(MAPFILE) --no-standard-libraries \
	-Wl,--retain-symbols-file,$(SYMFILE) \
	-Wl,--script,$(LDSCRIPT) $(IPA_FLAGS) $(LIBLDFLAGS)
	$(QUIET) $(OBJCOPY) --keep-global-symbols=$(SYMFILE) $@ $(TEMPOBJ)
	$(QUIET) $(OBJCOPY) --strip-unneeded $(TEMPOBJ) $@
	$(QUIET) -$(RM) $(TEMPOBJ)
endif 


$(OBJ_LIBO2OBJS): $(OBJDIR)/%.o: %.c
	@echo "Compiling $<"
	$(QUIET) $(CC) -o $@ $(OPT_O2) $(CFLAGS) $(INCLUDES) -c $<
	
$(OBJ_LIBOSOBJS): $(OBJDIR)/%.o: %.c
	@echo "Compiling $<"
	$(QUIET) $(CC) -o $@ $(OPT_OS) $(CFLAGS) $(INCLUDES) -c $<
	
$(OBJ_LIBO2CPPOBJS): $(OBJDIR)/%.o: %.cpp
	@echo "Compiling $<"
	$(QUIET) $(CXX) -o $@ $(OPT_O0) $(CFLAGS) -std=c++11 $(INCLUDES) -c $<
	
$(OBJ_LIBO2CCOBJS): $(OBJDIR)/%.o: %.cc
	@echo "Compiling $<"
	$(QUIET) $(CXX) -o $@ $(OPT_O0) $(CFLAGS) -std=c++11 $(INCLUDES) -c $<
	
$(OBJ_LIBOSCPPOBJS): $(OBJDIR)/%.o: %.cpp
	@echo "Compiling $<"
	$(QUIET) $(CXX) -o $@ $(OPT_O0) $(CFLAGS) -std=c++11 $(INCLUDES) -c $<
	
$(LIB): %.a: $(OBJDIR)/%.o
	@echo "Creating Library $@"
	$(QUIET) $(AR) rc $@ $^


ifeq ($(HOSTTYPE),win)
clean:
	-$(IFEXIST) xa_$(CODEC_NAME)$(DETECTED_CORE).a $(RM) xa_$(CODEC_NAME)$(DETECTED_CORE).a 
	-$(IFEXIST) xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(RM) xgcc_$(CODEC_NAME)$(DETECTED_CORE).a 
	-$(IFEXIST) $(LIBDIR)$(S)xa_$(CODEC_NAME)$(DETECTED_CORE).a $(RM) $(LIBDIR)$(S)xa_$(CODEC_NAME)$(DETECTED_CORE).a
	-$(IFEXIST) $(LIBDIR)$(S)xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(RM) $(LIBDIR)$(S)xgcc_$(CODEC_NAME)$(DETECTED_CORE).a
	-$(IFEXIST) $(MAPFILE) $(RM) $(MAPFILE)
	-$(IFEXIST) $(OBJDIR)$(S)*.o $(RM) $(OBJDIR)$(S)*.o
	-$(IFEXIST) $(OBJDIR)$(S)*.d $(RM) $(OBJDIR)$(S)*.d
	-$(IFEXIST) $(LIBDIR)  $(RM_R) $(LIBDIR)
else
clean:
	-$(RM) xa_$(CODEC_NAME)$(DETECTED_CORE).a xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(LIBDIR)$(S)xa_$(CODEC_NAME)$(DETECTED_CORE).a $(LIBDIR)$(S)xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(MAPFILE)
	-$(RM) $(OBJDIR)$(S)*.o
	-$(RM) $(ALL_DEPS)
	-$(RM_R) $(LIBDIR)
endif
