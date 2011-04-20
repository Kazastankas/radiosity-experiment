
TARGET = radiosity
TARGET_TYPE = executable

SRCS = \
	imageio.cpp \
	application.cpp \
	scene.cpp \
	main.cpp

CUDA_SRCS = \
	radiosity.cu

COMPILER_FLAGS =
LINK_LIBRARIES = -lGL -lGLU -lSDLmain -lSDL -lpng -lz -lcudart

