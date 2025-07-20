#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QTableWidget>
#include <vector>
#include "inference/kan_engine.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onRunClicked();
    void onResetClicked();

private:
    QWidget *centralWidget;
    QVBoxLayout *mainLayout;
    QHBoxLayout *inputLayout;
    QLabel *modelInfoLabel;
    QPushButton *runButton;
    QPushButton *resetButton;
    QTableWidget *outputTable;

    std::vector<QDoubleSpinBox*> inputFields;
    KANEngine* engine_;

    void setupUI();
    void populateInputs(int inputDim);
    void runKANInference();
};

#endif // MAINWINDOW_H
