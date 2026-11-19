#include "facesamplestorage.h"
#include "apppaths.h"
#include "imageutils.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QUuid>
#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace {

QString storageDirectory()
{
    return AppPaths::faceSampleDirectory();
}

}

cv::Mat FaceSampleStorage::loadImage(const QString &filePath,
                                     QString *errorMessage)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = QObject::tr("无法读取图片：%1").arg(filePath);
        }
        return {};
    }

    const QByteArray bytes = file.readAll();
    const auto *begin = reinterpret_cast<const unsigned char *>(bytes.constData());
    const std::vector<unsigned char> encoded(begin, begin + bytes.size());
    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty() && errorMessage) {
        *errorMessage = QObject::tr("图片格式无法解析：%1").arg(filePath);
    }
    return image;
}

QString FaceSampleStorage::saveAlignedFace(const cv::Mat &alignedFace,
                                           QString *errorMessage)
{
    if (alignedFace.empty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("没有可保存的对齐人脸。 ");
        }
        return {};
    }

    QDir directory;
    const QString targetDirectory = storageDirectory();
    if (!directory.mkpath(targetDirectory)) {
        if (errorMessage) {
            *errorMessage = QObject::tr("无法创建人脸样本目录：%1")
                                .arg(targetDirectory);
        }
        return {};
    }

    const QString fileName = QString("sample_%1.jpg")
                                 .arg(QUuid::createUuid().toString(QUuid::WithoutBraces));
    const QString filePath = QDir(targetDirectory).filePath(fileName);
    if (!ImageUtils::writeJpeg(filePath, alignedFace, 95, errorMessage)) {
        return {};
    }
    return QDir(AppPaths::dataDirectory()).relativeFilePath(filePath);
}

QString FaceSampleStorage::resolveStoredPath(const QString &storedPath)
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

bool FaceSampleStorage::deleteStoredImage(const QString &filePath)
{
    if (filePath.isEmpty()) {
        return true;
    }

    const QString currentRoot = QDir::fromNativeSeparators(
        QDir::cleanPath(storageDirectory())) + "/";
    const QString legacyRoot = QDir::fromNativeSeparators(
        QDir(AppPaths::applicationDirectory()).filePath(
            QStringLiteral("face_samples"))) + "/";
    const QString target = QDir::fromNativeSeparators(
        resolveStoredPath(filePath));
    if (!target.startsWith(currentRoot, Qt::CaseInsensitive)
        && !target.startsWith(legacyRoot, Qt::CaseInsensitive)) {
        return false;
    }
    return !QFile::exists(target) || QFile::remove(target);
}
