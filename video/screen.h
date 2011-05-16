#ifndef SCREEN_H
#define SCREEN_H

#include <GL/glew.h>

#include <QGLWidget>

#include "opencv2/opencv.hpp"
using namespace cv;

class Timer;

class Screen : public QGLWidget
{
    Q_OBJECT
public:
    explicit Screen(QWidget *parent = 0);
    void openTrimap(QString path);
    void openVideo(QString path);
    void openPhoto(QString path);
    void captureCamera();
    void release();
    void start();
    void stop();

private slots:
    void paintEvent(QPaintEvent *);
    void updateTexture();

protected:
    // opengl
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    // texture: opengl & cuda
    void createTexture();
    void releaseTexture();
    // cuda
    void runCUDA();

private:
    // display
    QTimer *timer;
    GLfloat x,y,s;
    // OpenCV
    Mat frame;
    Mat trimap;
    VideoCapture video;
    double last_time;
    double video_time;
    // OpenGL
    GLuint PBO;
    GLfloat ratio;
};

#endif // SCREEN_H
