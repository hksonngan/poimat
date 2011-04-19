#ifndef GUI_H
#define GUI_H

#include <QMainWindow>

class Screen;

namespace Ui {
    class Gui;
}

class Gui : public QMainWindow
{
    Q_OBJECT

public:
    explicit Gui(QWidget *parent = 0);
    ~Gui();

private slots:
    void exit();
    void openVideo();
    void openPhoto();
    void captureCamera();
    void mattingMode(bool);
    void play();
    void pause();

private:
    Ui::Gui *ui;
    Screen* screen;
};

#endif // GUI_H
