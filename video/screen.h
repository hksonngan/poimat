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
    void start();
    void stop();

private slots:
    void updateFrame();

protected:
    // ui
    void mouseMoveEvent(QMouseEvent *event);
    // opengl
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    // opengl
    void updateTexture(int index);
    void releaseTexture(int index);
    // cuda
    void initCUDA(int width, int height);
    void runCUDA();

private:
    // display
    QTimer *timer;
    bool playing;
    GLfloat x,y,s;
    Mode mode;

    // trimap
    bool newTrimap;
    struct keyPoint {
        int x;
        int y;
    };
    keyPoint forePoints[1024];
    keyPoint backPoints[1024];
    int foreTimestamps[1024];
    int backTimestamps[1024];
    int foreCount;
    int backCount;

    // OpenCV
    Mat frame;
    Mat trimap;
    Mat background;
    VideoCapture video;
    double last_time;
    double video_time;

    // OpenGL
    GLuint  PBO[2];
    GLuint  index;
    GLuint  textures[2];
    GLfloat ratio;
};

#endif // SCREEN_H
