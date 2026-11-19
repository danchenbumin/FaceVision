#include "apppaths.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>

namespace {

QString writableRoot()
{
    const QString besideExecutable = QDir(QCoreApplication::applicationDirPath())
                                        .filePath(QStringLiteral("data"));
    QDir directory;
    if (directory.mkpath(besideExecutable)) {
        QFileInfo info(besideExecutable);
        if (info.isWritable()) {
            return QDir::cleanPath(besideExecutable);
        }
    }

    const QString fallback = QStandardPaths::writableLocation(
        QStandardPaths::AppDataLocation);
    directory.mkpath(fallback);
    return QDir::cleanPath(fallback);
}

}

QString AppPaths::applicationDirectory()
{
    return QDir::cleanPath(QCoreApplication::applicationDirPath());
}

QString AppPaths::dataDirectory()
{
    static const QString directory = writableRoot();
    return directory;
}

QString AppPaths::databaseFile()
{
    const QString target = QDir(dataDirectory()).filePath(QStringLiteral("faces.db"));
    const QString legacy = QDir(applicationDirectory()).filePath(QStringLiteral("faces.db"));
    if (!QFileInfo::exists(target) && QFileInfo::exists(legacy)) {
        QFile::copy(legacy, target);
    }
    return target;
}

QString AppPaths::modelDirectory()
{
    return QDir(applicationDirectory()).filePath(QStringLiteral("models"));
}

QString AppPaths::faceSampleDirectory()
{
    return QDir(dataDirectory()).filePath(QStringLiteral("face_samples"));
}

QString AppPaths::recognitionSnapshotDirectory()
{
    return QDir(dataDirectory()).filePath(QStringLiteral("recognition_snapshots"));
}

bool AppPaths::ensureDataDirectories(QString *errorMessage)
{
    QDir directory;
    const QStringList paths{
        dataDirectory(),
        faceSampleDirectory(),
        recognitionSnapshotDirectory()
    };
    for (const QString &path : paths) {
        if (!directory.mkpath(path)) {
            if (errorMessage) {
                *errorMessage = QObject::tr("无法创建数据目录：%1").arg(path);
            }
            return false;
        }
    }
    return true;
}
