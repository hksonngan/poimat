#include "screen.h"
#include "cuda/filter.h"

#include <cutil_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include <QMouseEvent>
#include <QMessageBox>
#include <QDateTime>
#include <QTimer>
#include <QDebug>

Screen::Screen(QWidget *parent) :
    QGLWidget(parent)
{
    PBO[0] = PBO[1] = 0;

    x = 0;
    y = 0;
    s = 1;
    mode = NONE;

    newTrimap = false;

    playing = false;
    timer = new QTimer(this);
    timer->setInterval(100);
    timer->start();
    connect(timer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

void Screen::start()
{
    if (mode == VIDEO) {
        last_time = QDateTime::currentMSecsSinceEpoch();
    }
    if (mode == VIDEO || mode == CAMERA) {
        playing = true;
    }
}

void Screen::stop()
{
    if (mode == VIDEO || mode == CAMERA) {
        playing = false;
    }
}

void Screen::initializeGL()
{
    glewInit();

    glClearColor(0.85f, 0.85f, 0.85f, 0.0f);

    glEnable(GL_POINT_SMOOTH);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

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
        GLfloat w = r > 1.0f ? 1.0f/r : 1.0f;
        GLfloat h = r < 1.0f ? r : 1.0f;

        glBindTexture(GL_TEXTURE_2D, textures[1]);

        glBegin(GL_TRIANGLE_FAN);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(x-s*w, y+s*h);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(x+s*w, y+s*h);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(x+s*w, y-s*h);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(x-s*w, y-s*h);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (frame.data) {
        GLfloat r = (GLfloat)frame.rows/frame.cols/ratio;
        GLfloat w = r > 1.0f ? 1.0f/r : 1.0f;
        GLfloat h = r < 1.0f ? r : 1.0f;

        if (mode == CAMERA) {
            w *= -1.0f;
        }

        glBindTexture(GL_TEXTURE_2D, textures[0]);

        glBegin(GL_TRIANGLE_FAN);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(x-s*w, y+s*h);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(x+s*w, y+s*h);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(x+s*w, y-s*h);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(x-s*w, y-s*h);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);

        glPointSize(16.0f/frame.rows*this->width());

        glBegin(GL_POINTS);
            glColor3f(0.0f, 0.0f, 0.0f);
            for (int i=0; i<backCount; i++) {
                float u = (float)backPoints[i].x/frame.cols;
                float v = (float)backPoints[i].y/frame.rows;
                glVertex3f((u-0.5f)*2.0f*fabs(w), (0.5f-v)*2.0f*h, 0.1f);
            }
        glEnd();

        glBegin(GL_POINTS);
            glColor3f(1.0f, 1.0f, 1.0f);
            for (int i=0; i<foreCount; i++) {
                float u = (float)forePoints[i].x/frame.cols;
                float v = (float)forePoints[i].y/frame.rows;
                glVertex3f((u-0.5f)*2.0f*fabs(w), (0.5f-v)*2.0f*h, 0.1f);
            }
        glEnd();
    }

    glFlush();
}

void Screen::mouseMoveEvent(QMouseEvent *event)
{
    if (frame.data) {
        float r = (float)frame.rows/frame.cols/ratio;
        float w = r > 1.0f ? 1.0f/r : 1.0f;
        float h = r < 1.0f ? r : 1.0f;

        int x = frame.cols*(((float)event->x()/this->width()-0.5f)/w+0.5f);
        int y = frame.rows*(((float)event->y()/this->height()-0.5f)/h+0.5f);

        if(x<0 || x>=frame.cols || y<0 || y>=frame.rows)
            return;

        if (event->buttons() & Qt::LeftButton && foreCount < 1024) {
            forePoints[foreCount].x = x;
            forePoints[foreCount].y = y;
            foreTimestamps[foreCount] = QDateTime::currentMSecsSinceEpoch()+1000;
            foreCount++;
        }
        if (event->buttons() & Qt::RightButton && backCount < 1024) {
            backPoints[backCount].x = x;
            backPoints[backCount].y = y;
            backTimestamps[backCount] = QDateTime::currentMSecsSinceEpoch()+1000;
            backCount++;
        }
    }
}

void Screen::updateFrame()
{
    if (!playing) {
        updateGL();
        return;
    }
    if(video.get(CV_CAP_PROP_POS_FRAMES) > video.get(CV_CAP_PROP_FRAME_COUNT)){
        stop();
        return;
    }
    // update frame
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
    // update key points
    if (mode == VIDEO || mode == CAMERA) {
        int current = QDateTime::currentMSecsSinceEpoch();
        int count = 0;
        for (int i=0; i<foreCount; i++) {
            if (current < foreTimestamps[i]) {
                if (count != i) {
                    forePoints[count] = forePoints[i];
                    foreTimestamps[count] = foreTimestamps[i];
                }
                count++;
            }
        }
        foreCount = count;
        count = 0;
        for (int i=0; i<backCount; i++) {
            if (current < backTimestamps[i]) {
                if (count != i) {
                    backPoints[count] = backPoints[i];
                    backTimestamps[count] = backTimestamps[i];
                }
                count++;
            }
        }
        backCount = count;
    }
    updateTexture(0);
}

void Screen::releaseTexture(int index)
{
    if (index == 0) {
        releaseFilter();
        foreCount = 0;
        backCount = 0;
        if (PBO[0]) {
            CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(PBO[0]));
            glDeleteBuffersARB(1, &PBO[0]);
            PBO[0] = 0;
        }
        if (PBO[1]) {
            CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(PBO[1]));
            glDeleteBuffersARB(1, &PBO[1]);
            PBO[1] = 0;
        }
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
    if (index == 1) {
        if (background.data) {
            background.release();
        }
    }
}

void Screen::updateTexture(int index)
{
    if (index == 0) {
        if(trimap.data) {
            runCUDA();
        } else {
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, frame.data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }
    if (index == 1) {
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, background.cols, background.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, background.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    updateGL();
}

void Screen::initCUDA(int width, int height)
{
    GLsizei size = width*height*4*sizeof(float);

    // opengl
    glGenBuffersARB(2, PBO);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO[0]);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_STREAM_READ);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO[1]);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_STREAM_READ);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    index = 0;

    // cuda
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(PBO[0]));
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(PBO[1]));
    initializeFilter(width, height);
}

void Screen::runCUDA()
{
    Mat image;
    cvtColor(frame, image, CV_BGR2RGBA);

    unsigned int timer;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    if (!newTrimap) {
        trimapFilter(image.data, trimap.data, (int*)forePoints, foreCount, (int*)backPoints, backCount);
    }
    newTrimap = false;

    void *buffer;
    CUDA_SAFE_CALL(cudaGLMapBufferObject(&buffer, PBO[index]));
    poissonFilter(image.data, trimap.data, (float*)buffer);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObjectAsync(PBO[index], 0));

    cutilCheckError(cutStopTimer(timer));
    qDebug() << cutGetTimerValue(timer);
    cutilCheckError(cutDeleteTimer(timer));

    if (mode == VIDEO || mode == CAMERA)
        index = 1-index;

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBO[index]);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.cols, image.rows, 0, GL_RGBA, GL_FLOAT, 0);
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
    newTrimap = true;
    if (mode == PHOTO) {
        updateTexture(0);
    }
}

void Screen::openBackground(QString path)
{
    releaseTexture(1);
    background = imread(path.toStdString());
    if (!background.data) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the background file.");
        return;
    }
    updateTexture(1);
}

void Screen::openPhoto(QString path)
{
    releaseTexture(0);
    frame = imread(path.toStdString());
    if (!frame.data) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the image file.");
        return;
    }
    mode = PHOTO;
    updateTexture(0);
    initCUDA(frame.cols, frame.rows);
}

void Screen::openVideo(QString path)
{
    releaseTexture(0);
    video.open(path.toStdString());
    if (!video.isOpened()) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open the video file.");
        return;
    }
    mode = VIDEO;
    video_time = 0;
    initCUDA(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT));
    start();
}

void Screen::openCamera()
{
    releaseTexture(0);
    video.open(0);
    if (!video.isOpened()) {
        QMessageBox::warning(this, "Poisson Matting", "Can't open your camera.");
        return;
    }
    mode = CAMERA;
    initCUDA(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT));
    start();
}
