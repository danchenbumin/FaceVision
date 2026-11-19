#include "mainwindow.h"

#include <QApplication>
#include <QCoreApplication>

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("EngineeringTraining"));
    QCoreApplication::setApplicationName(QStringLiteral("FaceVision"));
    QCoreApplication::setApplicationVersion(QStringLiteral("2.0"));

    MainWindow window;
    window.show();
    return QApplication::exec();
}
