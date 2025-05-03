#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QSlider>

#include "inference/kan_engine.h"


class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onInputChanged(int value);

private:
    QWidget *centralWidget;
    QVBoxLayout *layout;
    QLabel *inputLabel;
    QSlider *inputSlider;
    QLabel *outputLabel;

    void setupUI();
    void runKANInference(float input);
    KANEngine* engine_;  // Add this member
};
};

#endif // MAINWINDOW_H
