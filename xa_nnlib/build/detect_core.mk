

ISS = xt-run $(XTCORE)
CONFIGDIR := $(shell $(ISS) --show-config=config)
include $(CONFIGDIR)/misc/hostenv.mk

GREPARGS =
ifeq ($(HOSTTYPE),win)
GREPARGS = /c:
endif

ifeq ("", "$(detected_core)")

hifi5="0"
hifi4="0"
hifi3z="0"
hifi3="0"
hifi1="0"

#simple logic to differentiate cores; need to optimize this logic
hifi5_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseHiFi5 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
hifi4_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseHiFi4 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
hifi3z_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseHiFi3Z = 1" "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
hifi3_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseHiFi3 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
hifi1_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseHiFi1 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")

#check exclusively for hifi5 or hifi4
    ifneq ("", "$(hifi5_tmp)")
        detected_core=hifi5
    else
        ifneq ("", "$(hifi4_tmp)")
            detected_core=hifi4
        else
            ifneq ("", "$(hifi3z_tmp)")
                detected_core=hifi3z
            else
                ifneq ("", "$(hifi1_tmp)")
                    detected_core=hifi1
                else
                    ifneq ("", "$(hifi3_tmp)")
                        detected_core=hifi3
                    endif
                endif
            endif
        endif
    endif

endif

ifeq ("$(detected_core)", "hifi5")
    hifi5=1
    CFLAGS+= -DCORE_HIFI5=1
else
    ifeq ("$(detected_core)", "hifi4")
        hifi4=1
        CFLAGS+= -DCORE_HIFI4=1
    else
        ifeq ("$(detected_core)", "hifi3z")
            hifi3z=1
            detected_core=hifi3z
            CFLAGS+= -DCORE_HIFI3Z=1
        else
            ifeq ("$(detected_core)", "hifi3")
                hifi3=1
                detected_core=hifi3
                CFLAGS+= -DCORE_HIFI3=1
            else
                ifeq ("$(detected_core)", "hifi1")
                      hifi1=1
                      hifi4=1
                      detected_core=hifi4
                      CFLAGS+= -DCORE_HIFI4=1
                else
                    ifeq ("$(detected_core)", "ref")
                        ref=1
                        CFLAGS+= -DREF_GCC=1
                    else
                        $(error Core Not Found)
                    endif
                endif
            endif
        endif
    endif
endif

xclib_tmp:=$(shell $(GREP) $(GREPARGS)"SW_CLibrary = xclib"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
ifneq ("", "$(xclib_tmp)")
    xclib=1
else
    xclib=0
endif

