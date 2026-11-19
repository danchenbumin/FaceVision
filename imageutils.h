#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <QString>
#include <opencv2/core.hpp>

namespace ImageUtils {

bool writeJpeg(const QString &filePath, const cv::Mat &image,
               int quality = 95, QString *errorMessage = nullptr);

}

#endif // IMAGEUTILS_H
