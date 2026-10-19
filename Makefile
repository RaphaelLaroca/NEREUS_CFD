

############################################################################################################
# NEREUS build. Linux, Windows (MSYS2 MINGW64 shell) and macOS.
#
#   make                       the GUI program, bin/nereus
#   make HEADLESS=1            bin/nereus_headless, no window code at all
#   make EIGEN_PATH=/path      use another Eigen (3.3 or newer): the folder that holds Eigen/Dense
#
# Everything else is in the repository: Eigen, ImGui/ImPlot/ImPlot3D, GLFW (compiled from include/glfw-3.4),
# tinyexpr and AMGCL. A Windows or macOS build needs only the compiler; see README.md, "Building".
############################################################################################################

# ---- settings you may want to change ------------------------------------------------------------------------
EIGEN_PATH ?= include/eigen-3.3.7  # Eigen root (the folder with Eigen/Dense in it); the copy in the repository by default
GLFW       ?= auto           # auto | vendored | system. auto: the system GLFW on Linux when pkg-config finds it, else
                             #   include/glfw-3.4 (always on Windows and macOS)
OPENMP     ?= 1              # 0 builds a serial program
NVIDIA_GPU ?= auto           # 1: AMGX + CUDA pressure solver, Linux only. auto: on when both are found below
HEADLESS   ?= 0              # 1: no GUI compiled or linked
DEBUG      ?= 0              # 1: -O0 -g with assertions; the default is -O3 -DNDEBUG
AMGX_PATH  ?= $(HOME)/SDK/AMGX
CUDA_PATH  ?= /usr/local/cuda
# --------------------------------------------------------------------------------------------------------------

# Strip the comments' trailing blanks off the values above.
EIGEN_PATH := $(strip $(EIGEN_PATH))
GLFW := $(strip $(GLFW))
OPENMP := $(strip $(OPENMP))
NVIDIA_GPU := $(strip $(NVIDIA_GPU))
HEADLESS := $(strip $(HEADLESS))
DEBUG := $(strip $(DEBUG))

############################
# Target system: LINUX (default), WIN or MAC. Windows exports OS=Windows_NT to every process, so a plain `make`
# in MSYS2 builds WIN; macOS is recognised by uname. `make OS=...` overrides both.
############################
ifeq ($(OS),Windows_NT)
OS := WIN
else ifneq ($(origin OS),command line)
ifeq ($(shell uname -s 2>/dev/null),Darwin)
OS := MAC
else
OS := LINUX
endif
endif
ifneq ($(filter $(OS),LINUX WIN MAC),$(OS))
$(error OS must be LINUX, WIN or MAC, not '$(OS)')
endif

# AMGX and CUDA are Linux only; auto turns them on when the AMGX library and the CUDA headers are where the
# paths above say.
ifneq ($(OS),LINUX)
override NVIDIA_GPU := 0
endif
ifeq ($(NVIDIA_GPU),auto)
ifneq ($(wildcard $(AMGX_PATH)/build/libamgxsh.so),)
ifneq ($(wildcard $(CUDA_PATH)/include/cuda_runtime.h),)
NVIDIA_GPU := 1
endif
endif
endif
ifneq ($(NVIDIA_GPU),1)
override NVIDIA_GPU := 0
endif

ifeq ($(DEBUG),1)
OPTFLAGS := -O0 -g -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS
else
OPTFLAGS := -O3 -DNDEBUG
endif

AMGX_INCLUDE_PATH := $(AMGX_PATH)/include
AMGX_BUILD_PATH := $(AMGX_PATH)/build
CUDA_LIB_PATH := $(CUDA_PATH)/lib64

IMGUI_DIR := include/imgui-1.92.4
IMPLOT_DIR := include/implot-0.17
IMPLOT3D_DIR := include/implot3d-0.3
TINYEXPR_DIR := include/tinyexpr
GLFW_DIR := include/glfw-3.4

############################
# Compilers: GCC on Linux and Windows (never MSVC), Clang on macOS. CXX=... CC=... on the command line wins.
############################
ifeq ($(OS),MAC)
DEFAULT_CXX := clang++
DEFAULT_CC := clang
else
DEFAULT_CXX := g++
DEFAULT_CC := gcc
endif
ifeq ($(origin CXX),default)
CXX := $(DEFAULT_CXX)
endif
ifeq ($(origin CC),default)
CC := $(DEFAULT_CC)
endif
NVCC := nvcc
IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -qi clang && echo 1)

############################
# C++ flags
############################
CXXFLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy
CXXFLAGS += $(OPTFLAGS) -MMD -MP
CXXFLAGS += -Isrc/headers -Isrc/headers/Solvers
CXXFLAGS += -I$(EIGEN_PATH)
CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
CXXFLAGS += -I$(IMPLOT_DIR)
CXXFLAGS += -I$(IMPLOT3D_DIR)
CXXFLAGS += -I$(TINYEXPR_DIR)
CXXFLAGS += -idirafter include/amgcl   # after the system headers: on a case-insensitive filesystem (macOS) <version> would
                                       # otherwise find our pin file include/amgcl/VERSION
CFLAGS := $(OPTFLAGS) -Wall -MMD -MP

ifeq ($(wildcard $(EIGEN_PATH)/Eigen/Dense),)
$(error Eigen not found: '$(EIGEN_PATH)/Eigen/Dense' does not exist. Set EIGEN_PATH to the Eigen root, e.g. make EIGEN_PATH=/usr/include/eigen3)
endif

############################
# OpenMP. Apple's clang has no OpenMP runtime of its own: Homebrew's libomp is used when it is installed
# (brew install libomp), and the program is built serial otherwise.
############################
OMPFLAGS :=
OMPLIBS :=
ifeq ($(OPENMP),1)
ifeq ($(OS)$(IS_CLANG),MAC1)
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(wildcard $(LIBOMP_PREFIX)/include/omp.h),)
OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
OMPLIBS := -L$(LIBOMP_PREFIX)/lib -lomp
else
$(warning libomp not found (brew install libomp): building without OpenMP, single threaded)
override OPENMP := 0
endif
else
OMPFLAGS := -fopenmp
OMPLIBS := -fopenmp
endif
endif
ifneq ($(OPENMP),1)
OMPFLAGS := -Wno-unknown-pragmas
endif
CXXFLAGS += $(OMPFLAGS)

############################
# GLFW, the one windowing library. OpenGL itself comes with the operating system, and ImGui's OpenGL 3 backend
# loads its own functions, so there is no loader to install.
############################
ifeq ($(GLFW),auto)
ifeq ($(OS),LINUX)
GLFW := $(shell pkg-config --exists glfw3 2>/dev/null && echo system || echo vendored)
else
GLFW := vendored
endif
endif
ifneq ($(filter $(GLFW),vendored system),$(GLFW))
$(error GLFW must be auto, vendored or system, not '$(GLFW)')
endif

GLFW_COMMON := context.c init.c input.c monitor.c platform.c vulkan.c window.c egl_context.c osmesa_context.c \
               null_init.c null_monitor.c null_window.c null_joystick.c
ifeq ($(OS),WIN)
GLFW_PLATFORM := -D_GLFW_WIN32
GLFW_FILES := $(GLFW_COMMON) win32_module.c win32_time.c win32_thread.c win32_init.c win32_joystick.c \
              win32_monitor.c win32_window.c wgl_context.c
else ifeq ($(OS),MAC)
GLFW_PLATFORM := -D_GLFW_COCOA
GLFW_FILES := $(GLFW_COMMON) cocoa_time.c posix_module.c posix_thread.c cocoa_init.m cocoa_joystick.m \
              cocoa_monitor.m cocoa_window.m nsgl_context.m
else
# X11; the libraries are opened at run time, the headers (libx11-dev, libxrandr-dev, libxinerama-dev,
# libxcursor-dev, libxi-dev) are needed to compile.
GLFW_PLATFORM := -D_GLFW_X11
GLFW_FILES := $(GLFW_COMMON) posix_time.c posix_module.c posix_thread.c posix_poll.c x11_init.c x11_monitor.c \
              x11_window.c xkb_unicode.c glx_context.c linux_joystick.c
endif
GLFW_CFLAGS := $(OPTFLAGS) -w -MMD -MP $(GLFW_PLATFORM) -I$(GLFW_DIR)/include

############################
# Per-OS switches and system libraries
############################
ifeq ($(OS),MAC)
CXXFLAGS += -DGL_SILENCE_DEPRECATION
SYSLIBS := -lpthread
GUILIBS := -framework OpenGL -framework Cocoa -framework IOKit -framework CoreFoundation -framework QuartzCore
else ifeq ($(OS),WIN)
# M_PI is not in MinGW's <cmath> under -std=c++17; Eigen's template objects overflow the COFF section limit.
CXXFLAGS += -D_USE_MATH_DEFINES -Wa,-mbig-obj
# Static, so the .exe runs outside MSYS2 with no DLL beside it.
SYSLIBS := -static
GUILIBS := -lopengl32 -lgdi32
else
SYSLIBS := -lrt -lpthread -ldl
GUILIBS := -lGL
endif

ifeq ($(GLFW),system)
CXXFLAGS += $(shell pkg-config --cflags glfw3 2>/dev/null)
GUILIBS := $(shell pkg-config --libs glfw3 2>/dev/null || echo -lglfw) $(GUILIBS)
else
CXXFLAGS += -I$(GLFW_DIR)/include
endif

############################
# NVCC flags
############################
NVCCFLAGS := -std=c++17 $(OPTFLAGS) -Xcompiler -Wall,-Wextra -MMD -MP
NVCCFLAGS += -Isrc/headers -Isrc/headers/Solvers
NVCCFLAGS += -I$(CUDA_PATH)/include
NVCCFLAGS += -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
NVCCFLAGS += -I$(EIGEN_PATH)

############################
# Linker flags
############################
LDFLAGS :=
ifeq ($(NVIDIA_GPU),1)
CXXFLAGS += -DNVIDIA_GPU -I$(CUDA_PATH)/include -I$(AMGX_INCLUDE_PATH) -I$(AMGX_BUILD_PATH)
LDFLAGS += -L$(CUDA_LIB_PATH) -L$(AMGX_BUILD_PATH)
LDFLAGS += -lamgx -lamgxsh -lmpi
LDFLAGS += -lcudart -lcublas -lcusparse -lcusolver
LDFLAGS += -lcudadevrt -lcudart_static
endif
LDFLAGS += $(OMPLIBS) $(SYSLIBS)
ifeq ($(HEADLESS),1)
CXXFLAGS += -DHEADLESS
else
LDFLAGS += $(GUILIBS)
endif

############################
# Directories
############################
SRC_DIR := src
CUDA_SRC_DIR := $(SRC_DIR)/cuda
BUILD_DIR := build
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
BIN_DIR := bin

############################
# Sources
############################
# Each method's solver files live in src/Methods/<method>/. Their objects are built flat into build/ beside the
# library's (the test runners link build/*.o), so a file name must be unique across all of src/.
METHOD_DIRS := $(wildcard $(SRC_DIR)/Methods/*)
SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(foreach d,$(METHOD_DIRS),$(wildcard $(d)/*.cpp))
vpath %.cpp $(SRC_DIR) $(METHOD_DIRS)
CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)

IMGUI_SRCS := \
  $(IMGUI_DIR)/imgui.cpp \
  $(IMGUI_DIR)/imgui_draw.cpp \
  $(IMGUI_DIR)/imgui_tables.cpp \
  $(IMGUI_DIR)/imgui_widgets.cpp \
  $(IMGUI_DIR)/misc/cpp/imgui_stdlib.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

IMPLOT_SRCS := \
  $(IMPLOT_DIR)/implot.cpp \
  $(IMPLOT_DIR)/implot_items.cpp

# The user interface (window, setup screen, viewer): GUI builds only, built into build/ui/ so the test runners' build/*.o
# glob, which links no ImGui, never picks it up.
UI_SRCS := $(wildcard $(SRC_DIR)/UI/*.cpp)

IMPLOT3D_SRCS := \
  $(IMPLOT3D_DIR)/implot3d.cpp \
  $(IMPLOT3D_DIR)/implot3d_items.cpp \
  $(IMPLOT3D_DIR)/implot3d_meshes.cpp

############################
# Objects
############################
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SRCS)))
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_BUILD_DIR)/%.o,$(CUDA_SRCS))

IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp,$(BUILD_DIR)/imgui/%.o,$(IMGUI_SRCS))
IMPLOT_OBJS := $(patsubst $(IMPLOT_DIR)/%.cpp,$(BUILD_DIR)/implot/%.o,$(IMPLOT_SRCS))
IMPLOT3D_OBJS := $(patsubst $(IMPLOT3D_DIR)/%.cpp,$(BUILD_DIR)/implot3d/%.o,$(IMPLOT3D_SRCS))
UI_OBJS := $(patsubst $(SRC_DIR)/UI/%.cpp,$(BUILD_DIR)/ui/%.o,$(UI_SRCS))

TINYEXPR_OBJS := $(BUILD_DIR)/tinyexpr/tinyexpr.o
GLFW_OBJS := $(addprefix $(BUILD_DIR)/glfw/,$(addsuffix .o,$(basename $(GLFW_FILES))))
ifneq ($(GLFW),vendored)
GLFW_OBJS :=
endif

ifneq ($(NVIDIA_GPU),1)
CUDA_OBJS :=
endif

ifeq ($(HEADLESS),1)
IMGUI_OBJS :=
IMPLOT_OBJS :=
IMPLOT3D_OBJS :=
GLFW_OBJS :=
UI_OBJS :=
endif

# Switching OS, compiler, NVIDIA_GPU, HEADLESS, DEBUG, OPENMP or GLFW rebuilds everything, so an object compiled in
# another mode cannot be linked.
MODE_STAMP := $(BUILD_DIR)/.mode_$(OS)_$(notdir $(firstword $(CXX)))_gpu$(NVIDIA_GPU)_headless$(HEADLESS)_debug$(DEBUG)_omp$(OPENMP)_glfw$(GLFW)

ALL_OBJS := $(OBJS) $(CUDA_OBJS) $(IMGUI_OBJS) $(IMPLOT_OBJS) $(IMPLOT3D_OBJS) $(TINYEXPR_OBJS) $(GLFW_OBJS) $(UI_OBJS)

############################
# Dependencies
############################
DEPS := $(ALL_OBJS:.o=.d)

############################
# Target
############################
ifeq ($(HEADLESS),1)
TARGET := $(BIN_DIR)/nereus_headless
else
TARGET := $(BIN_DIR)/nereus
endif
ifeq ($(OS),WIN)
TARGET := $(TARGET).exe
endif

all: $(TARGET)

############################
# Directories
############################
$(BUILD_DIR) $(CUDA_BUILD_DIR) \
$(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends $(BUILD_DIR)/imgui/misc/cpp \
$(BUILD_DIR)/implot $(BUILD_DIR)/implot3d $(BUILD_DIR)/tinyexpr $(BUILD_DIR)/glfw $(BUILD_DIR)/ui \
$(BIN_DIR):
	mkdir -p $@

############################
# Link
############################
$(TARGET): $(ALL_OBJS) | $(BIN_DIR)
	$(CXX) $(ALL_OBJS) -o $@ $(LDFLAGS)

############################
# Compile rules
############################
$(MODE_STAMP): | $(BUILD_DIR)
	rm -f $(BUILD_DIR)/.nvidia_gpu_* $(BUILD_DIR)/.mode_*
	touch $@

$(BUILD_DIR)/%.o: %.cpp $(MODE_STAMP) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ui/%.o: $(SRC_DIR)/UI/%.cpp $(MODE_STAMP) | $(BUILD_DIR)/ui
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_BUILD_DIR)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/imgui/%.o: $(IMGUI_DIR)/%.cpp $(MODE_STAMP) | $(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends $(BUILD_DIR)/imgui/misc/cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/imgui/backends/%.o: $(IMGUI_DIR)/backends/%.cpp $(MODE_STAMP) | $(BUILD_DIR)/imgui/backends
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/implot/%.o: $(IMPLOT_DIR)/%.cpp $(MODE_STAMP) | $(BUILD_DIR)/implot
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/implot3d/%.o: $(IMPLOT3D_DIR)/%.cpp $(MODE_STAMP) | $(BUILD_DIR)/implot3d
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tinyexpr/%.o: $(TINYEXPR_DIR)/%.c $(MODE_STAMP) | $(BUILD_DIR)/tinyexpr
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/glfw/%.o: $(GLFW_DIR)/src/%.c $(MODE_STAMP) | $(BUILD_DIR)/glfw
	@mkdir -p $(@D)
	$(CC) $(GLFW_CFLAGS) -c $< -o $@

$(BUILD_DIR)/glfw/%.o: $(GLFW_DIR)/src/%.m $(MODE_STAMP) | $(BUILD_DIR)/glfw
	@mkdir -p $(@D)
	$(CC) $(GLFW_CFLAGS) -c $< -o $@

############################
# Run
############################
run: $(TARGET)
	./$(TARGET)

# What the build will use, without building.
info:
	@echo "OS=$(OS) CXX=$(CXX) CC=$(CC) clang=$(if $(IS_CLANG),yes,no)"
	@echo "EIGEN_PATH=$(EIGEN_PATH) GLFW=$(GLFW) OPENMP=$(OPENMP) NVIDIA_GPU=$(NVIDIA_GPU) HEADLESS=$(HEADLESS) DEBUG=$(DEBUG)"
	@echo "TARGET=$(TARGET)"

############################
# Clean
############################
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all run info clean

