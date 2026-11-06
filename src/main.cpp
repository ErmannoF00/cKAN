// Retained original Qt application entry point; CLI lives in src/cli_main.cpp.
#include <QApplication>
#include "gui/MainWindow.h"
int main(int argc,char** argv) {
    QApplication app(argc,argv);
    MainWindow window;
    window.show();
    return app.exec();
}
