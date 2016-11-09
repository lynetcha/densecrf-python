CC = gcc
CFLAGS = -std=c++11 -fPIC -Wall 
LDFLAGS = -shared # -Wl,--no-allow-shlib-undefined
DEBUG_FLAGS = -g -DEBUG
RELEASE_FLAGS = -O2 -D NDEBUG

# ANACONDA_PATH = /Users/chrischoy/anaconda/
ANACONDA_PATH = /scr/lynetcha/programs/anaconda/envs/3dds/

# Run numpy.get_include() in python to get the following path
NUMPY_INCLUDE = $(ANACONDA_PATH)/lib/python2.7/site-packages/numpy/core/include

# PYCONFIG_PATH = /usr/local/python2.7/
PYCONFIG_PATH = $(ANACONDA_PATH)/include/python2.7

TARGET   = Pydensecrf.so
# SOURCES  = wrapper/Pydensecrf.cpp
SOURCES  = src/permutohedral.cpp src/pairwise.cpp src/util.cpp src/unary.cpp src/labelcompatibility.cpp src/objective.cpp src/optimization.cpp src/densecrf.cpp wrapper/Pydensecrf.cpp
INCLUDES = -I./include -I./src -I$(PYCONFIG_PATH) -I$(NUMPY_INCLUDE) -I./external/liblbfgs/include
M_LIBRARY_PATH = -L/usr/local/lib -L./external

# LIBS     = -framework OpenGL -lOpenThreads -losg -losgDB -losgGA -losgViewer -losgUtil -lstdc++ -lm -lboost_python -lpython2.7
#LIBS     = -lGL -lGLU -losg -losgDB -losgGA -losgViewer -losgUtil -lstdc++ -lm -lboost_python -lpython2.7
LIBS     =  -lstdc++ -lboost_python -lpython2.7 ./external/liblbfgs.a

all: clean
	$(CC) $(CFLAGS) $(LDFLAGS) $(RELEASE_FLAGS) $(SOURCES) -o $(TARGET) $(INCLUDES) $(M_LIBRARY_PATH) $(LIBS)

debug:
	$(CC) $(CFLAGS) $(LDFLAGS) $(DEBUG_FLAGS) $(SOURCES) -o $(TARGET) $(INCLUDES) $(M_LIBRARY_PATH) $(LIBS)

clean:
	rm -f $(TARGET)
