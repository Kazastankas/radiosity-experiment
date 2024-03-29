
# Authors: Eric Butler
# DO NOT EDIT THIS FILE

# default to debug
MODE ?= debug

include build/defines.mk

.PHONY: all radiosity clean

all: radiosity

radiosity:
	$(MAKE) -f build/make.mk MODE=$(MODE) MKFILE=build/radiosity.mk target
	cp $(BIN_DIR)/$(MODE)/radiosity .

clean:
	rm -rf radiosity $(OBJ_DIR) $(BIN_DIR)

