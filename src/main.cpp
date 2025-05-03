#include <QApplication>
#include "gui/MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow window;
    window.setWindowTitle("KAN Inference Viewer");
    window.resize(800, 600);
    window.show();

    return app.exec();
}
