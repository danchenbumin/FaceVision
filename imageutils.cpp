#include "imageutils.h"

#include <QFile>
#include <QObject>
#include <opencv2/imgcodecs.hpp>
#include <vector>

bool ImageUtils::writeJpeg(const QString &filePath, const cv::Mat &image,
                           int quality, QString *errorMessage)
{
    if (image.empty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("没有可保存的图片数据。");
        }
        return false;
    }

    std::vector<unsigned char> encoded;
    if (!cv::imencode(".jpg", image, encoded,
                      {cv::IMWRITE_JPEG_QUALITY, quality})) {
        if (errorMessage) {
            *errorMessage = QObject::tr("OpenCV 图片编码失败。");
        }
        return false;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (errorMessage) {
            *errorMessage = QObject::tr("无法写入图片：%1").arg(filePath);
        }
        return false;
    }
    const qint64 written = file.write(
        reinterpret_cast<const char *>(encoded.data()),
        static_cast<qint64>(encoded.size()));
    if (written != static_cast<qint64>(encoded.size())) {
        if (errorMessage) {
            *errorMessage = QObject::tr("图片写入不完整：%1").arg(filePath);
        }
        return false;
    }
    return true;
}
