#include "MainWindow.h"
#include "model_loader.h"
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setupUI();
    setWindowTitle("cKAN | validated inference viewer");
    resize(860,620);
}
void MainWindow::setupUI() {
    auto* central=new QWidget(this);
    mainLayout_=new QVBoxLayout(central);
    auto* top=new QHBoxLayout;
    auto* load=new QPushButton("Open model (.kan / .json)",this);
    modelInfoLabel_=new QLabel("No model loaded",this);
    top->addWidget(load);
    top->addWidget(modelInfoLabel_,1);
    mainLayout_->addLayout(top);
    connect(load,&QPushButton::clicked,this,[this]() {
        const QString path=QFileDialog::getOpenFileName(this,"Load cKAN model",QString(),"cKAN models (*.kan *.json)");
        if(!path.isEmpty()) {
            try {loadPath(path);}
            catch(const std::exception& e) {QMessageBox::critical(this,"Model rejected",e.what());}
        }
    });
    backend_=new QComboBox(this);
    backend_->addItems({"baseline","shared","packed"});
    backend_->setCurrentText("packed");
    mainLayout_->addWidget(backend_);
    connect(backend_,&QComboBox::currentTextChanged,this,[this](const QString&) {
       if(model_) {
          try {executor_=std::make_unique<kan::Executor>(*model_,kan::parse_engine(backend_->currentText().toStdString()));}
          catch(const std::exception& e) {QMessageBox::critical(this,"Backend failed",e.what());}
       }
    });
    auto* inputHost=new QWidget(this);
    inputLayout_=new QFormLayout(inputHost);
    mainLayout_->addWidget(inputHost);
    auto* actions=new QHBoxLayout;
    auto* run=new QPushButton("Run inference",this);
    auto* reset=new QPushButton("Reset inputs",this);
    actions->addWidget(run);actions->addWidget(reset);
    mainLayout_->addLayout(actions);
    connect(run,&QPushButton::clicked,this,&MainWindow::onRunClicked);
    connect(reset,&QPushButton::clicked,this,&MainWindow::onResetClicked);
    outputTable_=new QTableWidget(this);
    outputTable_->setColumnCount(2);
    outputTable_->setHorizontalHeaderLabels({"Output", "Value"});
    outputTable_->horizontalHeader()->setStretchLastSection(true);
    mainLayout_->addWidget(outputTable_);
    setCentralWidget(central);
}
void MainWindow::populateInputs(std::size_t dimension) {
    if(dimension>64) throw std::invalid_argument("GUI accepts up to 64 inputs; use kan_infer CLI for wider models");
    while(inputLayout_->rowCount()>0) inputLayout_->removeRow(0);
    inputFields_.clear();
    for(std::size_t i=0;i<dimension;++i) {
        auto* spin=new QDoubleSpinBox(this);
        spin->setDecimals(6);
        spin->setRange(-100000.0,100000.0);
        spin->setSingleStep(0.1);
        inputLayout_->addRow("x["+QString::number(static_cast<qulonglong>(i))+"]",spin);
        inputFields_.push_back(spin);
    }
    outputTable_->setRowCount(0);
}
void MainWindow::loadPath(const QString& path) {
    const auto native=path.toStdString();
    // Construct new model first. If load fails, previous model remains usable.
    if(path.endsWith(".kan",Qt::CaseInsensitive)) {
        auto next=std::make_unique<kan::Model>(kan::load_model(native));
        if(next->layers.front().inputs>64) throw std::runtime_error("GUI input width >64; use CLI");
        auto nextEngine=std::make_unique<kan::Executor>(*next,kan::parse_engine(backend_->currentText().toStdString()));
        populateInputs(next->layers.front().inputs);
        model_=std::move(next);
        executor_=std::move(nextEngine);
        legacy_.reset();
        family_=Family::EdgeSpline;
        backend_->setEnabled(true);
        modelInfoLabel_->setText("CKAN 1 (edge-spline) | "+path);
    } else if(path.endsWith(".json",Qt::CaseInsensitive)) {
#ifdef CKAN_HAS_JSON
        auto next=std::make_unique<KANEngine>(loadKANModelFromJSON(native));
        if(next->model().input_dim>64) throw std::runtime_error("GUI input width >64; use kan_legacy_infer CLI");
        populateInputs(static_cast<std::size_t>(next->model().input_dim));
        legacy_=std::move(next);
        executor_.reset();model_.reset();
        family_=Family::LegacyJSON;
        backend_->setEnabled(false);
        modelInfoLabel_->setText("Legacy W2 phi(W1 x) | "+path);
#else
        throw std::runtime_error("Legacy JSON disabled: install libboost-dev and rebuild");
#endif
    } else throw std::invalid_argument("select a .kan or .json model");
}
void MainWindow::onRunClicked() {
  try {
    if(family_==Family::None) throw std::runtime_error("Load a model first");
    std::vector<float> x;
    x.reserve(inputFields_.size());
    for(auto* field:inputFields_) x.push_back(static_cast<float>(field->value()));
    std::vector<float> result;
    if(family_==Family::LegacyJSON) result=legacy_->infer(x);
    else result=executor_->forward(x);
    outputTable_->setRowCount(static_cast<int>(result.size()));
    for(std::size_t i=0;i<result.size();++i) {
      outputTable_->setItem(static_cast<int>(i),0,new QTableWidgetItem(QString::number(static_cast<qulonglong>(i))));
      outputTable_->setItem(static_cast<int>(i),1,new QTableWidgetItem(QString::number(result[i],'g',9)));
    }
  } catch(const std::exception& e) {QMessageBox::warning(this,"Inference error",e.what());}
}
void MainWindow::onResetClicked() {
    for(auto* v:inputFields_) v->setValue(0.0);
    outputTable_->setRowCount(0);
}
