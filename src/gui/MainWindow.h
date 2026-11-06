#pragma once
#include <QMainWindow>
#include <QString>
#include <memory>
#include <vector>
#include "kan/model.hpp"
#include "inference/kan_engine.h"
class QComboBox;
class QLabel;
class QPushButton;
class QFormLayout;
class QDoubleSpinBox;
class QTableWidget;
class QVBoxLayout;
class QWidget;

// Original GUI entry point, now connected to either the legacy JSON or CKAN 1 model.
class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);
    ~MainWindow() override = default;
private:
    enum class Family { None, LegacyJSON, EdgeSpline };
    Family family_ = Family::None;
    std::unique_ptr<KANEngine> legacy_;
    std::unique_ptr<kan::Model> model_;
    std::unique_ptr<kan::Executor> executor_;
    QVBoxLayout* mainLayout_ = nullptr;
    QFormLayout* inputLayout_ = nullptr;
    QComboBox* backend_ = nullptr;
    QLabel* modelInfoLabel_ = nullptr;
    QTableWidget* outputTable_ = nullptr;
    std::vector<QDoubleSpinBox*> inputFields_;
    void setupUI();
    void populateInputs(std::size_t dimension);
    void loadPath(const QString& path);
    void onRunClicked();
    void onResetClicked();
};
