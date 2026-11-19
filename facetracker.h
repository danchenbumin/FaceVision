#ifndef FACETRACKER_H
#define FACETRACKER_H

#include "facerecognizer.h"

#include <QHash>
#include <QVector>
#include <opencv2/core.hpp>
#include <vector>

struct TrackedFace
{
    int trackId = -1;
    cv::Rect smoothedBox;
};

class FaceTracker
{
public:
    QVector<TrackedFace> update(const std::vector<FaceDetection> &detections);
    void reset();

private:
    struct TrackState {
        int id = -1;
        cv::Rect2f box;
        int missedFrames = 0;
    };

    static float intersectionOverUnion(const cv::Rect2f &a,
                                       const cv::Rect2f &b);

    QVector<TrackState> tracks;
    int nextTrackId = 1;
    static constexpr int MaxMissedFrames = 5;
    static constexpr float MatchIouThreshold = 0.25f;
    static constexpr float SmoothingFactor = 0.65f;
};

#endif // FACETRACKER_H
