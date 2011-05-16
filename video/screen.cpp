#include "screen.h"
#include "filter.h"

#include <cutil_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

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

    timer = new QTimer(this);
    timer->setInterval(100);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateTexture()));
}

void Screen::initializeGL()
{
    glewInit();

    glClearColor(0.85f, 0.85f, 0.85f, 0.0f);
    glEnable(GL_TEXTURE_2D);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void Screen::resizeGL(int w, int h)
{
    glViewport(0, 0, (GLint)w, (GLint)h);
    ratio = (GLfloat)h/w;
}

void Screen::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLfloat r = (frame.data ? (GLfloat)frame.rows/frame.cols : ratio) / ratio;
    GLfloat h = r < 1.0f ? r : 1.0f;
    GLfloat w = r > 1.0f ? 1.0f/r : 1.0f;

    glBegin(GL_TRIANGLE_FAN);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(x-s*w, y+s*h);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(x+s*w, y+s*h);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(x+s*w, y-s*h);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(x-s*w, y-s*h);
    glEnd();

    glFlush();
}

void Screen::runCUDA()
{
    Mat image;
    cvtColor(frame, image, CV_BGR2RGBA);

    void *buffer;
    CUDA_SAFE_CALL(cudaGLMapBufferObject(&buffer, PBO));
    if (trimap.data) {
        qDebug() << "poissonFilter";
        poissonFilter(image.data, trimap.data, (float*)buffer, image.cols, image.rows);
    } else {
        cudaMemcpy(buffer, image.data, image.step*image.rows, cudaMemcpyHostToDevice);
    }
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(PBO));

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);
    if (trimap.data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGBA, GL_FLOAT, 0);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void Screen::paintEvent(QPaintEvent *)
{
    updateGL();
}

void Screen::start()
{
    if (video.isOpened()) {
        last_time = QDateTime::currentMSecsSinceEpoch();
        timer->start();
    }
}

void Screen::stop()
{
    if (video.isOpened()) {
        timer->stop();
    }
}

void Screen::createTexture()
{
    GLsizei size = frame.cols*frame.rows*4*sizeof(float);

    glGenBuffersARB(1, &PBO);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_STREAM_READ);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(PBO));

    initializeTexture(frame.cols, frame.rows);

    video_time = 0;

    qDebug() << frame.cols << "x" << frame.rows;
}

void Screen::releaseTexture()
{
    if (PBO) {
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(PBO));
        glDeleteBuffersARB(1, &PBO);
        PBO = 0;
    }
}

void Screen::updateTexture()
{
    runCUDA();
    updateGL();
    if (video.isOpened()) {
        double current_time = QDateTime::currentMSecsSinceEpoch();
        video_time += current_time-last_time;
        last_time = current_time;
        while (video_time > video.get(CV_CAP_PROP_POS_MSEC)) {
            video >> frame;
        }
    }
}

void Screen::openTrimap(QString path)
{
    trimap = imread(path.toStdString(), 0);
    updateTexture();
}

void Screen::openVideo(QString path)
{
    releaseTexture();
    release();
    video.open(path.toStdString());
    if (!video.isOpened()) {
        qDebug() << "!video.isOpened()";
        return;
    }
    video >> frame;
    createTexture();
    start();
}

void Screen::openPhoto(QString path)
{
    releaseTexture();
    release();
    frame = imread(path.toStdString());
    createTexture();
    updateTexture();
}

void Screen::captureCamera()
{
    releaseTexture();
    release();
    video.open(0);
    if (!video.isOpened()) {
        qDebug() << "!video.isOpened()";
        return;
    }
    video >> frame;
    createTexture();
    start();
}

void Screen::release()
{
    if (trimap.data) {
        trimap.release();
    }
    if (video.isOpened()) {
        video.release();
    }
    stop();
}
