# simple-annotation-tool
A simple annotation tool in C++, also a simple OpenGL demo of drawing rectangles and polygons.

### In Windows with vs2013
add headers and libs in folder opencv, freeglut and glew-2.0.0


### In Ubuntu
OpenCV and OpenGL should be installed
build: ``g++ draw.cpp -o draw `pkg-config --cflags --libs opencv` -lglut -lGLU -lGL -lm -lGLEW``
