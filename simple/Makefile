CC = c++
CC_FLAGS = -O3 -g -Wall -ffast-math

CC_SRCS = driver.cc Instance.cc timer.cc checks.cc \
	best_improvement.cc swap.cc

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
