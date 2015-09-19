CC=g++
CFLAGS=-c -g -Wall -O0
LDFLAGS= -lm -lpthread -lX11 -L/usr/X11R6/lib64
SOURCES=main.cpp input.cpp geometric.cpp drawing.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=run

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) *.o *~ $(EXECUTABLE)