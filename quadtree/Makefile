CXX = g++
CXX_FLAGS = -O3 -g -Wall -std=c++11 -pedantic

SRCS = driver.cc \
	Instance.cc Tour.cc MortonKey.cc Quadtree.cc QuadtreeNode.cc \
	TreeOpt.cc Segment.cc

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
