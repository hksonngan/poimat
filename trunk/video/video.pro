#-------------------------------------------------
#
# Project created by QtCreator 2011-04-13T18:28:00
#
#-------------------------------------------------

QT += core gui
QT += opengl

TARGET = video
TEMPLATE = app

SOURCES += \
    main.cpp\
    gui.cpp \
    screen.cpp

HEADERS += \
    gui.h \
    screen.h \
    filter.h \
    filter.h

FORMS += \
    gui.ui

RESOURCES += \
    resource.qrc

#-------------------------------------------------
#   CUDA
#-------------------------------------------------

CUDA_SOURCES = \
    poisson.cu\
    filter.cu

win32 {
  INCLUDEPATH += $(CUDA_INC_PATH) $(CUTIL_INC_PATH)
  QMAKE_LIBDIR += $(CUDA_LIB_PATH) $(CUTIL_LIB_PATH)
  LIBS += cudart.lib cutil32D.lib glew32.lib cufft.lib
  QMAKE_LFLAGS += /NODEFAULTLIB:libcmt

  cuda.output = $$OBJECTS_DIR${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = nvcc.exe \
    --machine 32 -ccbin \"$(VCINSTALLDIR)/bin\" -maxrregcount=16 --ptxas-options=-v --compile \
    -Xcompiler \"/EHsc /W3 /nologo /O2 /Zi /MT\" $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}
unix {
  # auto-detect CUDA path
  CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
  INCLUDEPATH += $$CUDA_DIR/include
  QMAKE_LIBDIR += $$CUDA_DIR/lib
  LIBS += -lcudart

  cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
  cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda

#-------------------------------------------------
#   OpenCV
#-------------------------------------------------

INCLUDEPATH += \
    C:/OpenCV2.2/include

QMAKE_LIBDIR += \
    C:/OpenCV2.2/lib

LIBS += \
    opencv_calib3d220d.lib \
    opencv_contrib220d.lib \
    opencv_core220d.lib \
    opencv_features2d220d.lib \
    opencv_ffmpeg220d.lib \
    opencv_flann220d.lib \
    opencv_gpu220d.lib \
    opencv_highgui220d.lib \
    opencv_imgproc220d.lib \
    opencv_legacy220d.lib \
    opencv_ml220d.lib \
    opencv_objdetect220d.lib \
    opencv_ts220.lib \
    opencv_video220d.lib
