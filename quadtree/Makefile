CC = g++
CC_FLAGS = -O3 -g -Wall -std=c++11 -pedantic

CC_SRCS = driver.cc checks.cc \
	best_improvement.cc swap.cc \
	morton_serial.cc Node.cc \
	quadtree_serial.cc segment_serial.cc \
	input.cc

%.o: %.cc
	$(CC) $(CC_FLAGS) -o $@ -c $<

MY_OBJS = $(CC_SRCS:.cc=.o)

tsp: $(MY_OBJS)
	$(CC) $(CC_FLAGS) $^ -o tsp

CLEANFILES = tsp
CLEANFILES += $(MY_OBJS)
CLEANFILES += *.o[0-9][0-9]*
CLEANFILES += *.e[0-9][0-9]*
CLEANFILES += *.btr

clean:
	rm -f core *~ $(CLEANFILES)

# eof