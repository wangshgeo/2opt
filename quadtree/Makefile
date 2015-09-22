CXX = g++
CXX_FLAGS = -O3 -g -Wall -std=c++11 -pedantic

SRCS = driver.cc checks.cc \
	best_improvement.cc swap.cc \
	morton_serial.cc Node.cc \
	quadtree_serial.cc segment_serial.cc \
	input.cc \
	Instance.cc Tour.cc


%.o: %.cc
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

OBJS = $(SRCS:.cc=.o)

tsp: $(OBJS)
	$(CXX) $(CXX_FLAGS) $^ -o tsp

CLEANFILES = tsp
CLEANFILES += $(OBJS)
CLEANFILES += *.o[0-9][0-9]*
CLEANFILES += *.e[0-9][0-9]*
CLEANFILES += *.btr

clean:
	rm -f core *~ $(CLEANFILES)

# eof