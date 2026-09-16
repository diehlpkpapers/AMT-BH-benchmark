# Barnes-Hut on Charm++ -- AMT benchmark implementation.
#
# Point CHARM_PATH at a built Charm++ tree, or override it on the command line:
#   make CHARM_PATH=/path/to/charm/<target>
CHARM_PATH ?= $(HOME)/charm_reconverse

INCPATH  = $(CHARM_PATH)/include
CHARMC   = $(CHARM_PATH)/bin/charmc

# -DSTATISTICS reports interaction counts per iteration.
# -DCHECK_TRAVERSAL_MASS reports, per step and PE, the range of total mass each
# bucket interacted with. Exact traversal => every bucket sees the same figure,
# so any spread is a dropped or duplicated interaction. Off by default; it
# allocates a map entry per bucket.
# Other useful toggles: -DVERBOSE_TRAVERSAL -DCHECK_NUM_INTERACTIONS
#                       -DPRINT_TREE -DDEBUG_TRAVERSALS -DVERBOSE_NODE_REFINE
APP_FLAGS = -DSTATISTICS
OPTS      = -O3 -g $(APP_FLAGS)

# Charm++ tracing, off by default. The trace libraries instrument every entry
# method, so a traced binary is not the one the benchmark should be timed on --
# build it separately when you want to look inside a run.
#
#   make TRACE=projections   full event log per PE (.log + .sts), for the
#                            Projections GUI. Detailed and large; keep runs
#                            short (a couple of hundred steps) or the writes
#                            themselves distort what you are measuring.
#   make TRACE=summary       per-PE utilization over time (.sum). Far smaller,
#                            and enough to see load imbalance and idle time.
#   make                     no tracing.
#
# Use the -timers= phase report for routine measurements; reach for these when
# the phase numbers say something is wrong and you need to see where.
TRACE ?= none
ifneq ($(TRACE),none)
  TRACEMODE = -tracemode $(TRACE)
endif

CXXFLAGS += $(OPTS) -I$(INCPATH)
LDFLAGS  += $(OPTS) -L. -language charm++ -module RotateLB -memory os $(TRACEMODE)

CXX = $(CHARMC)
CC  = $(CXX)
AR  = ar q
CXX_DEPEND = $(CXX) -M -MM -MG -Wall $(APP_FLAGS)
CFLAGS = $(OPTS) $(DEFINE_FLAGS) -g

OBJECTS = Main.o DataManager.o TreePiece.o util.o Reduction.o Worker.o Request.o State.o
SRC = Main.cpp DataManager.cpp TreePiece.cpp \
      util.cpp Reduction.cpp Worker.cpp Request.cpp \
      State.cpp particleGenerator.cpp gen_util.cpp plummer.cpp

TARGET = barnes
all: $(TARGET) gen plummer

$(TARGET): $(OBJECTS) .tracemode
	$(CHARMC) -o $(TARGET) $(LDFLAGS) $(OBJECTS)

# -tracemode is a link-time choice only, so switching TRACE= would not
# otherwise rebuild anything. Record the current setting and relink when it
# changes.
.tracemode: FORCE
	@[ -f $@ ] && [ "`cat $@`" = "$(TRACE)" ] || { echo "$(TRACE)" > $@; }

FORCE:

# Standalone input generators. These are plain C++; they do not link Charm++,
# but they do include the shared headers, so they need the Charm++ include path
# for pup.h.
gen: particleGenerator.cpp
	$(CXX) $(OPTS) -I$(INCPATH) -o gen particleGenerator.cpp

plummer.o: plummer.cpp
	$(CXX) $(OPTS) -I$(INCPATH) -c plummer.cpp

gen_util.o: gen_util.cpp
	$(CXX) $(OPTS) -I$(INCPATH) -c gen_util.cpp

plummer: plummer.o gen_util.o
	$(CXX) $(OPTS) -o plummer plummer.o gen_util.o

%.decl.h %.def.h : %.ci
	$(CHARMC) $(APP_FLAGS) -E $<

%.o: Makefile

clean:
	rm -f core* $(OBJECTS) *~ $(TARGET) *.decl.h *.def.h charmrun conv-host \
	      gen plummer plummer.o gen_util.o .tracemode

# Header dependencies. Every translation unit includes barnes.decl.h, which
# charmc generates from barnes.ci via the pattern rule above and which a clean
# build cannot discover for itself. The wildcard is deliberately coarse -- any
# header change rebuilds all eight objects -- because the alternative was no
# header dependencies at all, which meant editing a .h quietly left the old
# object files in place. Eight files rebuild in seconds; a stale build costs a
# lot more than that.
HEADERS = $(wildcard *.h)
$(OBJECTS): barnes.decl.h $(HEADERS)
plummer.o gen_util.o: $(HEADERS)
gen: $(HEADERS)

# Finer-grained header dependencies are optional. `make depends` writes them to
# Makefile.dep, which is not committed because it records absolute paths into
# the Charm++ tree. `-include` tolerates its absence, and it is deliberately
# NOT auto-generated: the charmc -M pass writes stray object files, so letting
# make trigger it mid-build races with compilation.
depends:
	$(CXX_DEPEND) $(SRC) | while read i;do echo $$i| awk -F' ' '{for (i=1;i<NF;++i) print $$i" \\"}';echo;done|grep -v "$(CHARM_PATH)/bin" > Makefile.dep.tmp
	mv Makefile.dep.tmp Makefile.dep
	rm -f $(OBJECTS)

.PHONY: all clean depends FORCE

-include Makefile.dep
