#include "MainWindow.h"
#include <QString>
#include "model_loader.h"
#include "inference/kan_engine.h"
#include <QFile>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), engine_(nullptr) {
    setupUI();

    try {
        KANModel model = loadKANModelFromJSON("models/dummy_model.json");
        engine_ = new KANEngine(model);
    } catch (const std::exception& e) {
        qDebug() << "Failed to load KAN model:" << e.what();
    }
}

MainWindow::~MainWindow() {
    delete engine_;
}

void MainWindow::setupUI() {
    centralWidget = new QWidget(this);
    layout = new QVBoxLayout(centralWidget);

    inputLabel = new QLabel("Input Value: 0", this);
    inputSlider = new QSlider(Qt::Horizontal, this);
    inputSlider->setMinimum(-100);
    inputSlider->setMaximum(100);
    inputSlider->setValue(0);

    outputLabel = new QLabel("Output: N/A", this);

    layout->addWidget(inputLabel);
    layout->addWidget(inputSlider);
    layout->addWidget(outputLabel);

    setCentralWidget(centralWidget);

    connect(inputSlider, &QSlider::valueChanged, this, &MainWindow::onInputChanged);
}

void MainWindow::onInputChanged(int value) {
    inputLabel->setText("Input Value: " + QString::number(value));
    runKANInference(static_cast<float>(value) / 100.0f);
}

void MainWindow::runKANInference(float input) {
    if (!engine_) {
        outputLabel->setText("Error: KAN model not loaded.");
        return;
    }

    std::vector<float> inputVec = { input };
    std::vector<float> result = engine_->infer(inputVec);
    if (!result.empty()) {
        outputLabel->setText("Output: " + QString::number(result[0]));
    } else {
        outputLabel->setText("Output: (empty)");
    }
}
