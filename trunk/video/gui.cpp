#include "gui.h"
#include "ui_gui.h"

#include "screen.h"

#include <QDebug>

#include <QProgressBar>
#include <QSlider>
#include <QLabel>
#include <QFileDialog>

Gui::Gui(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Gui)
{
    ui->setupUi(this);

    ui->timelineToolBar->addWidget(new QSlider(Qt::Horizontal));

    ui->statusBar->addWidget(new QLabel("Ready"), 1);

    screen = new Screen(ui->centralWidget);
    ui->gridLayout->addWidget(screen, 0, 0, 1, 1);

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(exit()));
    connect(ui->actionVideo, SIGNAL(triggered()), this, SLOT(openVideo()));
    connect(ui->actionPhoto, SIGNAL(triggered()), this, SLOT(openPhoto()));
    connect(ui->actionCamera, SIGNAL(triggered()), this, SLOT(captureCamera()));
    connect(ui->actionMat, SIGNAL(triggered()), this, SLOT(matting()));
    connect(ui->actionPlay, SIGNAL(triggered()), this, SLOT(play()));
    connect(ui->actionPause, SIGNAL(triggered()), this, SLOT(pause()));
}

Gui::~Gui()
{
    delete ui;
    delete screen;
}

void Gui::exit()
{
    this->destroy();
}

void Gui::openVideo()
{
    QString path = QFileDialog::getOpenFileName(
        this, QString::null,  QString::null,
        "Vedios (*.avi *.mpg *.mp4)");

    if (path == QString::null) {
        qDebug() << "path == QString::null";
        return;
    }

    qDebug() << path;

    screen->openVideo(path);
}

void Gui::openPhoto()
{
    QString path = QFileDialog::getOpenFileName(
        this, QString::null,  QString::null,
        "Photos (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.pbm, *.pgm *.ppm *.sr *.ras *.tiff *.tif)"
        );

    if (path == QString::null) {
        qDebug() << "path == QString::null";
        return;
    }

    qDebug() << path;

    screen->openPhoto(path);
}

void Gui::captureCamera()
{
    screen->captureCamera();
}

void Gui::matting()
{
    QString path = QFileDialog::getOpenFileName(
        this, QString::null,  QString::null,
        "Photos (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.pbm, *.pgm *.ppm *.sr *.ras *.tiff *.tif)"
        );

    if (path == QString::null) {
        qDebug() << "path == QString::null";
        return;
    }

    qDebug() << path;

    screen->openTrimap(path);
}

void Gui::play()
{
    screen->start();
}

void Gui::pause()
{
    screen->stop();
}
