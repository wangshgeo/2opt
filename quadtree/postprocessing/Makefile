CC = c++
FLAGS = -O3 -g -Wall -lX11 -lpthread

SRCS = driver.cc image.cc
OBJS = $(SRCS:.cc=.o)

%.o: %.cc
	$(CC) -c $< -o $@ $(FLAGS) 

post: $(OBJS)
	$(CC) -o $@ $^ $(FLAGS) 

CLEANFILES = post
CLEANFILES += $(OBJS)
CLEANFILES += *.o[0-9][0-9]*
CLEANFILES += *.e[0-9][0-9]*
CLEANFILES += *.btr
# CLEANFILES += output/timestep*

clean:
	rm -f *~ $(CLEANFILES)

reset:
	rm -f *~ output/timestep*
