#ifndef FACEMATCHER_H
#define FACEMATCHER_H

#include "facedatabase.h"

#include <QString>
#include <QVector>
#include <memory>
#include <opencv2/core.hpp>

namespace cv::flann {
class Index;
}

struct FaceMatchResult
{
    int personId = -1;
    QString name;
    float similarity = 0.0f;
    float secondBestSimilarity = 0.0f;
};

class FaceMatcher
{
public:
    FaceMatcher();
    ~FaceMatcher();
    void rebuild(const QVector<PersonInfo> &persons);
    FaceMatchResult match(const cv::Mat &feature) const;

    int sampleCount() const;
    int personCount() const;
    QString searchModeName() const;

private:
    FaceMatchResult exactMatch(const cv::Mat &feature) const;
    FaceMatchResult approximateMatch(const cv::Mat &feature) const;
    void updateResult(FaceMatchResult &best, int personId,
                      const QString &name, float similarity) const;

    QVector<int> samplePersonIds;
    QVector<QString> samplePersonNames;
    cv::Mat sampleMatrix;

    QVector<int> centerPersonIds;
    QVector<QString> centerPersonNames;
    cv::Mat centerMatrix;

    std::unique_ptr<cv::flann::Index> flannIndex;
    bool approximateMode = false;
    static constexpr int ApproximateSearchThreshold = 200;
};

#endif // FACEMATCHER_H
