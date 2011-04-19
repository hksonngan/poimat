/********************************************************************************
** Form generated from reading UI file 'gui.ui'
**
** Created: Fri Apr 15 20:37:45 2011
**      by: Qt User Interface Compiler version 4.7.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GUI_H
#define UI_GUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Gui
{
public:
    QAction *actionExit;
    QAction *actionPlay;
    QAction *actionPause;
    QAction *actionVideo;
    QAction *actionPhoto;
    QAction *actionMat;
    QAction *actionCamera;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuPlay;
    QToolBar *playerToolBar;
    QToolBar *timelineToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *Gui)
    {
        if (Gui->objectName().isEmpty())
            Gui->setObjectName(QString::fromUtf8("Gui"));
        Gui->resize(400, 300);
        actionExit = new QAction(Gui);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionPlay = new QAction(Gui);
        actionPlay->setObjectName(QString::fromUtf8("actionPlay"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icons/38-airplane.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPlay->setIcon(icon);
        actionPause = new QAction(Gui);
        actionPause->setObjectName(QString::fromUtf8("actionPause"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/icons/48-fork-and-knife.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPause->setIcon(icon1);
        actionVideo = new QAction(Gui);
        actionVideo->setObjectName(QString::fromUtf8("actionVideo"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/icons/70-tv.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionVideo->setIcon(icon2);
        actionPhoto = new QAction(Gui);
        actionPhoto->setObjectName(QString::fromUtf8("actionPhoto"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/icons/42-photos.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPhoto->setIcon(icon3);
        actionMat = new QAction(Gui);
        actionMat->setObjectName(QString::fromUtf8("actionMat"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/icons/icons/183-genie-lamp.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionMat->setIcon(icon4);
        actionCamera = new QAction(Gui);
        actionCamera->setObjectName(QString::fromUtf8("actionCamera"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/icons/icons/86-camera.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCamera->setIcon(icon5);
        centralWidget = new QWidget(Gui);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        Gui->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(Gui);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuPlay = new QMenu(menuBar);
        menuPlay->setObjectName(QString::fromUtf8("menuPlay"));
        Gui->setMenuBar(menuBar);
        playerToolBar = new QToolBar(Gui);
        playerToolBar->setObjectName(QString::fromUtf8("playerToolBar"));
        playerToolBar->setMovable(false);
        playerToolBar->setIconSize(QSize(36, 24));
        playerToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        Gui->addToolBar(Qt::BottomToolBarArea, playerToolBar);
        timelineToolBar = new QToolBar(Gui);
        timelineToolBar->setObjectName(QString::fromUtf8("timelineToolBar"));
        timelineToolBar->setMovable(false);
        Gui->addToolBar(Qt::BottomToolBarArea, timelineToolBar);
        Gui->insertToolBarBreak(timelineToolBar);
        statusBar = new QStatusBar(Gui);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        Gui->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuPlay->menuAction());
        menuFile->addAction(actionVideo);
        menuFile->addAction(actionPhoto);
        menuFile->addAction(actionCamera);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);
        menuPlay->addAction(actionPlay);
        menuPlay->addAction(actionPause);
        playerToolBar->addAction(actionVideo);
        playerToolBar->addAction(actionPhoto);
        playerToolBar->addAction(actionCamera);
        playerToolBar->addSeparator();
        playerToolBar->addAction(actionMat);
        playerToolBar->addSeparator();
        playerToolBar->addAction(actionPlay);
        playerToolBar->addAction(actionPause);

        retranslateUi(Gui);

        QMetaObject::connectSlotsByName(Gui);
    } // setupUi

    void retranslateUi(QMainWindow *Gui)
    {
        Gui->setWindowTitle(QApplication::translate("Gui", "Poisson Matting", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("Gui", "Exit", 0, QApplication::UnicodeUTF8));
        actionPlay->setText(QApplication::translate("Gui", "Play", 0, QApplication::UnicodeUTF8));
        actionPause->setText(QApplication::translate("Gui", "Pause", 0, QApplication::UnicodeUTF8));
        actionVideo->setText(QApplication::translate("Gui", "Video", 0, QApplication::UnicodeUTF8));
        actionPhoto->setText(QApplication::translate("Gui", "Photo", 0, QApplication::UnicodeUTF8));
        actionMat->setText(QApplication::translate("Gui", "Mat", 0, QApplication::UnicodeUTF8));
        actionCamera->setText(QApplication::translate("Gui", "Camera", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("Gui", "File", 0, QApplication::UnicodeUTF8));
        menuPlay->setTitle(QApplication::translate("Gui", "Play", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Gui: public Ui_Gui {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUI_H
