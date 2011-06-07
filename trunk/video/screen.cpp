#include "screen.h"
#include "cuda/filter.h"

#include <cutil_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include <QMessageBox>
#include <QDateTime>
#include <QTimer>
#include <QDebug>

Screen::Screen(QWidget *parent) :
    QGLWidget(parent)
{
    PBO = 0;

    x = 0;
    y = 0;
    s = 1;
    mode = NONE;

    timer = new QTimer(this);
    timer->setInterval(50);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

void Screen::start()
{
    if (mode == VIDEO) {
        last_time = QDateTime::currentMSecsSinceEpoch();
    }
    if (mode == VIDEO || mode == CAMERA) {
        timer->start();
    }
}

void Screen::stop()
{
    if (mode == VIDEO || mode == CAMERA) {
        timer->stop();
    }
}

void Screen::initializeGL()
{
    glewInit();
    glClearColor(0.85f, 0.85f, 0.85f, 0.0f);
    glEnable(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glGenTextures(2, textures);

    glBindTexture(GL_TEXTURE_2D, textures[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, textures[1]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Screen::resizeGL(int w, int h)
{
    glViewport(0, 0, (GLint)w, (GLint)h);
    ratio = (GLfloat)h/w;
}

void Screen::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (background.data) {
        GLfloat r = (GLfloat)background.rows/background.cols/ratio;
        GLfloat h = r < 1.0f ? r : 1.0f;
        GLfloat w = r > 1.0f ? 1.0f/r : 1.0f;

        glBindTexture(GL_TEXTURE_2D, textures[1]);

        glBegin(GL_TRIANGLE_FAN);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(x-s*w, y+s*h);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(x+s*w, y+s*h);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(x+s*w, y-s*h);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(x-s*w, y-s*h);
        glEnd();
    }

    if (frame.data) {
        GLfloat r = (GLfloat)frame.rows/frame.cols/ratio;
        GLfloat h = r < 1.0f ? r : 1.0f;
        GLfloat w = r > 1.0f ? 1.0f/r : 1.0f;

        glBindTexture(GL_TEXTURE_2D, textures[0]);

        glBegin(GL_TRIANGLE_FAN);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(x-s*w, y+s*h);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(x+s*w, y+s*h);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(x+s*w, y-s*h);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(x-s*w, y-s*h);
        glEnd();
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    glFlush();
}

void Screen::updateFrame()
{
    if (mode == VIDEO) {
        double current_time = QDateTime::currentMSecsSinceEpoch();
        video_time += current_time-last_time;
        last_time = current_time;
        while (video_time > video.get(CV_CAP_PROP_POS_MSEC)) {
            video >> frame;
        }
    } else if (mode == CAMERA) {
        video >> frame;
    }
    updateTexture(0);
}

void Screen::createTexture(int width, int height)
{
    GLsizei size = width*height*4*sizeof(float);

    // opengl
    glGenBuffersARB(1, &PBO);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_STREAM_READ);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // cuda
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(PBO));
    initializeTexture(width, height);
}

void Screen::releaseTexture()
{
    if (PBO) {
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(PBO));
        glDeleteBuffersARB(1, &PBO);
        PBO = 0;
    }
}

void Screen::updateTexture(int index)
{
    if (index == 0) {
        if(trimap.data) {
            runCUDA();
        } else {
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, 3, frame.cols, frame.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, frame.data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }
    if (index == 1) {
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, 3, background.cols, background.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, background.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    updateGL();
}

void Screen::runCUDA()
{
    Mat image;
    cvtColor(frame, image, CV_BGR2RGBA);

    void *buffer;
    CUDA_SAFE_CALL(cudaGLMapBufferObject(&buffer, PBO));
    poissonFilter(image.data, trimap.data, (float*)buffer, image.cols, image.rows);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(PBO));

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void Screen::openTrimap(QString path)
{
    if (!frame.data) {
        QMessageBox::warning(this, "Poisson Matting", "Please open image/video/camera first.");
        return;
    }
    trimap = imread(path.toStdString(), 0);
    if (!trimap.data) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the trimap file.");
        return;
    }
    if (frame.cols != trimap.cols || frame.rows != trimap.rows) {
        trimap.release();
        QMessageBox::warning(this, "Poisson Matting", "Trimap size mismatch.");
        return;
    }
    if (mode == PHOTO) {
        updateTexture(0);
    }
}

void Screen::openBackground(QString path)
{
    background = imread(path.toStdString(), 0);
    if (!background.data) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the background file.");
        return;
    }
    updateTexture(1);
}

void Screen::openPhoto(QString path)
{
    release();
    frame = imread(path.toStdString());
    if (!frame.data) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the image file.");
        return;
    }
    mode = PHOTO;
    createTexture(frame.cols, frame.rows);
    updateTexture(0);
}

void Screen::openVideo(QString path)
{
    release();
    video.open(path.toStdString());
    if (!video.isOpened()) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the video file.");
        return;
    }
    mode = VIDEO;
    video_time = 0;
    createTexture(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT));
    start();
}

void Screen::openCamera()
{
    release();
    video.open(0);
    if (!video.isOpened()) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open your camera.");
        return;
    }
    mode = CAMERA;
    createTexture(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT));
    start();
}

void Screen::release()
{
    releaseTexture();
    if (trimap.data) {
        trimap.release();
    }
    if (background.data) {
        background.release();
    }
    if (frame.data) {
        frame.release();
    }
    if (video.isOpened()) {
        video.release();
        stop();
    }
    mode = NONE;
}
