#ifndef SCREEN_H
#define SCREEN_H

#include <GL/glew.h>

#include <QGLWidget>

#include "opencv2/opencv.hpp"
using namespace cv;

class Timer;

enum Mode {
    NONE,
    PHOTO,
    VIDEO,
    CAMERA,
};

class Screen : public QGLWidget
{
    Q_OBJECT
public:
    explicit Screen(QWidget *parent = 0);
    void openTrimap(QString path);
    void openBackground(QString path);
    void openVideo(QString path);
    void openPhoto(QString path);
    void openCamera();
    void release();
    void start();
    void stop();

private slots:
    void updateFrame();

protected:
    // opengl
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    // opengl
    void createTexture(int width, int height);
    void updateTexture(int index);
    void releaseTexture();
    // cuda
    void runCUDA();

private:
    // display
    QTimer *timer;
    GLfloat x,y,s;
    Mode mode;
    // OpenCV
    Mat frame;
    Mat trimap;
    Mat background;
    VideoCapture video;
    double last_time;
    double video_time;
    // OpenGL
    GLuint  PBO;
    GLuint  textures[2];
    GLfloat ratio;
};

#endif // SCREEN_H
