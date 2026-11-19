#include "facematcher.h"

#include <algorithm>
#include <cmath>
#include <opencv2/flann.hpp>
#include <vector>

FaceMatcher::FaceMatcher() = default;
FaceMatcher::~FaceMatcher() = default;

namespace {

cv::Mat normalizedRow(const cv::Mat &feature)
{
    if (feature.empty() || feature.type() != CV_32F || feature.total() != 128) {
        return {};
    }
    cv::Mat row = feature.reshape(1, 1).clone();
    cv::normalize(row, row, 1.0, 0.0, cv::NORM_L2);
    return row;
}

}

void FaceMatcher::rebuild(const QVector<PersonInfo> &persons)
{
    samplePersonIds.clear();
    samplePersonNames.clear();
    centerPersonIds.clear();
    centerPersonNames.clear();
    sampleMatrix.release();
    centerMatrix.release();
    flannIndex.reset();
    approximateMode = false;

    std::vector<cv::Mat> samples;
    std::vector<cv::Mat> centers;

    for (const PersonInfo &person : persons) {
        if (person.embeddings.isEmpty()) {
            continue;
        }

        cv::Mat center = cv::Mat::zeros(1, 128, CV_32F);
        int validSamples = 0;
        for (const cv::Mat &embedding : person.embeddings) {
            const cv::Mat row = normalizedRow(embedding);
            if (row.empty()) {
                continue;
            }
            samples.push_back(row);
            samplePersonIds.append(person.id);
            samplePersonNames.append(person.name);
            center += row;
            ++validSamples;
        }

        if (validSamples > 0) {
            center /= static_cast<float>(validSamples);
            cv::normalize(center, center, 1.0, 0.0, cv::NORM_L2);
            centers.push_back(center);
            centerPersonIds.append(person.id);
            centerPersonNames.append(person.name);
        }
    }

    if (!samples.empty()) {
        cv::vconcat(samples, sampleMatrix);
    }
    if (!centers.empty()) {
        cv::vconcat(centers, centerMatrix);
    }

    if (sampleMatrix.rows >= ApproximateSearchThreshold) {
        flannIndex = std::make_unique<cv::flann::Index>(
            sampleMatrix, cv::flann::KDTreeIndexParams(4));
        approximateMode = true;
    }
}

FaceMatchResult FaceMatcher::match(const cv::Mat &feature) const
{
    if (sampleMatrix.empty()) {
        return {};
    }
    return approximateMode ? approximateMatch(feature) : exactMatch(feature);
}

FaceMatchResult FaceMatcher::exactMatch(const cv::Mat &feature) const
{
    FaceMatchResult result;
    const cv::Mat query = normalizedRow(feature);
    if (query.empty()) {
        return result;
    }

    cv::Mat sampleScores = sampleMatrix * query.t();
    for (int row = 0; row < sampleScores.rows; ++row) {
        updateResult(result, samplePersonIds[row], samplePersonNames[row],
                     sampleScores.at<float>(row, 0));
    }

    if (!centerMatrix.empty()) {
        cv::Mat centerScores = centerMatrix * query.t();
        for (int row = 0; row < centerScores.rows; ++row) {
            updateResult(result, centerPersonIds[row], centerPersonNames[row],
                         centerScores.at<float>(row, 0));
        }
    }
    return result;
}

FaceMatchResult FaceMatcher::approximateMatch(const cv::Mat &feature) const
{
    FaceMatchResult result;
    const cv::Mat query = normalizedRow(feature);
    if (query.empty() || !flannIndex) {
        return result;
    }

    const int neighbours = std::min(8, sampleMatrix.rows);
    cv::Mat indices(1, neighbours, CV_32S);
    cv::Mat distances(1, neighbours, CV_32F);
    flannIndex->knnSearch(query, indices, distances, neighbours,
                          cv::flann::SearchParams(64));

    for (int column = 0; column < neighbours; ++column) {
        const int row = indices.at<int>(0, column);
        if (row < 0 || row >= sampleMatrix.rows) {
            continue;
        }
        const float squaredDistance = distances.at<float>(0, column);
        const float cosine = 1.0f - squaredDistance * 0.5f;
        updateResult(result, samplePersonIds[row], samplePersonNames[row], cosine);
    }

    if (!centerMatrix.empty()) {
        cv::Mat centerScores = centerMatrix * query.t();
        for (int row = 0; row < centerScores.rows; ++row) {
            updateResult(result, centerPersonIds[row], centerPersonNames[row],
                         centerScores.at<float>(row, 0));
        }
    }
    return result;
}

void FaceMatcher::updateResult(FaceMatchResult &best, int personId,
                               const QString &name, float similarity) const
{
    if (personId == best.personId) {
        best.similarity = std::max(best.similarity, similarity);
        return;
    }

    if (similarity > best.similarity) {
        best.secondBestSimilarity = best.similarity;
        best.similarity = similarity;
        best.personId = personId;
        best.name = name;
    } else if (similarity > best.secondBestSimilarity) {
        best.secondBestSimilarity = similarity;
    }
}

int FaceMatcher::sampleCount() const
{
    return sampleMatrix.rows;
}

int FaceMatcher::personCount() const
{
    return centerMatrix.rows;
}

QString FaceMatcher::searchModeName() const
{
    return approximateMode ? QObject::tr("KD树近似检索")
                           : QObject::tr("批量矩阵精确检索");
}
