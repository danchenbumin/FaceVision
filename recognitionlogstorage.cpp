#include "recognitionlogstorage.h"
#include "apppaths.h"
#include "imageutils.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QUuid>
#include <opencv2/imgcodecs.hpp>

namespace {

QString storageDirectory()
{
    return AppPaths::recognitionSnapshotDirectory();
}

}

QString RecognitionLogStorage::saveSnapshot(const cv::Mat &frame,
                                             const cv::Rect &faceBox,
                                             QString *errorMessage)
{
    if (frame.empty() || faceBox.empty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("没有可保存的识别抓拍。");
        }
        return {};
    }

    // Keep some surrounding context instead of saving only a tightly cropped face.
    const int horizontalPadding = faceBox.width / 4;
    const int verticalPadding = faceBox.height / 3;
    const cv::Rect expanded(faceBox.x - horizontalPadding,
                            faceBox.y - verticalPadding,
                            faceBox.width + horizontalPadding * 2,
                            faceBox.height + verticalPadding * 2);
    const cv::Rect safeBox = expanded & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.empty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("识别到的人脸区域超出图片范围。");
        }
        return {};
    }

    QDir directory;
    const QString targetDirectory = storageDirectory();
    if (!directory.mkpath(targetDirectory)) {
        if (errorMessage) {
            *errorMessage = QObject::tr("无法创建识别抓拍目录：%1")
                                .arg(targetDirectory);
        }
        return {};
    }

    const QString fileName = QString("log_%1.jpg")
                                 .arg(QUuid::createUuid().toString(
                                     QUuid::WithoutBraces));
    const QString absolutePath = QDir(targetDirectory).filePath(fileName);
    if (!ImageUtils::writeJpeg(absolutePath, frame(safeBox), 90, errorMessage)) {
        return {};
    }
    return QDir(AppPaths::dataDirectory()).relativeFilePath(absolutePath);
}

QString RecognitionLogStorage::resolveStoredPath(const QString &storedPath)
{
    if (storedPath.isEmpty()) {
        return {};
    }
    const QFileInfo info(storedPath);
    if (info.isAbsolute()) {
        return QDir::cleanPath(storedPath);
    }
    const QString currentPath = QDir::cleanPath(
        QDir(AppPaths::dataDirectory()).filePath(storedPath));
    if (QFileInfo::exists(currentPath)) {
        return currentPath;
    }
    const QString legacyPath = QDir::cleanPath(
        QDir(AppPaths::applicationDirectory()).filePath(storedPath));
    return QFileInfo::exists(legacyPath) ? legacyPath : currentPath;
}

bool RecognitionLogStorage::deleteStoredSnapshot(const QString &storedPath)
{
    if (storedPath.isEmpty()) {
        return true;
    }

    const QString currentRoot = QDir::fromNativeSeparators(
        QDir::cleanPath(storageDirectory())) + "/";
    const QString legacyRoot = QDir::fromNativeSeparators(
        QDir(AppPaths::applicationDirectory()).filePath(
            QStringLiteral("recognition_snapshots"))) + "/";
    const QString target = QDir::fromNativeSeparators(
        resolveStoredPath(storedPath));
    if (!target.startsWith(currentRoot, Qt::CaseInsensitive)
        && !target.startsWith(legacyRoot, Qt::CaseInsensitive)) {
        return false;
    }
    return !QFile::exists(target) || QFile::remove(target);
}
