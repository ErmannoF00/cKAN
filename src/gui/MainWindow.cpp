#include "MainWindow.h"
#include <QString>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include <QTableWidget>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QScrollArea>
#include <QDebug>
#include "model_loader.h"
#include "inference/kan_engine.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), engine_(nullptr) {
    setupUI();

    try {
        KANModel model = loadKANModelFromJSON("models/dummy_model.json");
        engine_ = new KANEngine(model);
        setupDynamicInputs(model.input_dim);
    } catch (const std::exception& e) {
        qDebug() << "Failed to load KAN model:" << e.what();
    }
}

MainWindow::~MainWindow() {
    delete engine_;
}

void MainWindow::setupUI() {
    centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // Model Info
    modelInfoLabel = new QLabel("Model: Not loaded", this);
    mainLayout->addWidget(modelInfoLabel);

    // Input panel
    inputLayout = new QFormLayout();
    QWidget *inputWidget = new QWidget(this);
    inputWidget->setLayout(inputLayout);
    inputFields.clear();
    mainLayout->addWidget(inputWidget);

    // Buttons
    QPushButton *runButton = new QPushButton("Run Inference", this);
    QPushButton *resetButton = new QPushButton("Reset", this);
    connect(runButton, &QPushButton::clicked, this, &MainWindow::onRunInference);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetInputs);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(runButton);
    buttonLayout->addWidget(resetButton);
    mainLayout->addLayout(buttonLayout);

    // Output
    outputTable = new QTableWidget(this);
    outputTable->setColumnCount(2);
    outputTable->setHorizontalHeaderLabels({"Index", "Value"});
    outputTable->horizontalHeader()->setStretchLastSection(true);
    mainLayout->addWidget(outputTable);

    setCentralWidget(centralWidget);
}

void MainWindow::setupDynamicInputs(int inputDim) {
    for (int i = 0; i < inputDim; ++i) {
        QDoubleSpinBox *spin = new QDoubleSpinBox(this);
        spin->setRange(-10.0, 10.0);
        spin->setSingleStep(0.1);
        spin->setValue(0.0);
        inputFields.push_back(spin);
        inputLayout->addRow("Input " + QString::number(i), spin);
    }
    modelInfoLabel->setText("Model loaded | Input Dim: " + QString::number(inputDim));
}

void MainWindow::onRunInference() {
    if (!engine_) return;

    std::vector<float> inputVec;
    for (QDoubleSpinBox *spin : inputFields) {
        inputVec.push_back(spin->value());
    }

    std::vector<float> output = engine_->infer(inputVec);
    outputTable->setRowCount(output.size());
    for (int i = 0; i < output.size(); ++i) {
        outputTable->setItem(i, 0, new QTableWidgetItem(QString::number(i)));
        outputTable->setItem(i, 1, new QTableWidgetItem(QString::number(output[i])));
    }
}

void MainWindow::onResetInputs() {
    for (QDoubleSpinBox *spin : inputFields) {
        spin->setValue(0.0);
    }
    outputTable->clearContents();
    outputTable->setRowCount(0);
}
