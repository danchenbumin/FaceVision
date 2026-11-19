#include "facetracker.h"

#include <algorithm>
#include <limits>

QVector<TrackedFace> FaceTracker::update(
    const std::vector<FaceDetection> &detections)
{
    QVector<TrackedFace> output;
    output.resize(static_cast<int>(detections.size()));
    QVector<bool> trackUsed(tracks.size(), false);

    for (int detectionIndex = 0;
         detectionIndex < static_cast<int>(detections.size());
         ++detectionIndex) {
        const cv::Rect2f detectionBox = detections[detectionIndex].box;
        int bestTrack = -1;
        float bestIou = MatchIouThreshold;

        for (int trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
            if (trackUsed[trackIndex]) {
                continue;
            }
            const float iou = intersectionOverUnion(
                tracks[trackIndex].box, detectionBox);
            if (iou > bestIou) {
                bestIou = iou;
                bestTrack = trackIndex;
            }
        }

        if (bestTrack < 0) {
            TrackState state;
            state.id = nextTrackId++;
            state.box = detectionBox;
            tracks.append(state);
            bestTrack = tracks.size() - 1;
            trackUsed.append(true);
        } else {
            trackUsed[bestTrack] = true;
            TrackState &state = tracks[bestTrack];
            state.box.x = state.box.x * SmoothingFactor
                          + detectionBox.x * (1.0f - SmoothingFactor);
            state.box.y = state.box.y * SmoothingFactor
                          + detectionBox.y * (1.0f - SmoothingFactor);
            state.box.width = state.box.width * SmoothingFactor
                              + detectionBox.width * (1.0f - SmoothingFactor);
            state.box.height = state.box.height * SmoothingFactor
                               + detectionBox.height * (1.0f - SmoothingFactor);
            state.missedFrames = 0;
        }

        const TrackState &state = tracks[bestTrack];
        output[detectionIndex] = {
            state.id,
            cv::Rect(cvRound(state.box.x), cvRound(state.box.y),
                     cvRound(state.box.width), cvRound(state.box.height))
        };
    }

    for (int index = tracks.size() - 1; index >= 0; --index) {
        if (index >= trackUsed.size() || !trackUsed[index]) {
            ++tracks[index].missedFrames;
            if (tracks[index].missedFrames > MaxMissedFrames) {
                tracks.removeAt(index);
            }
        }
    }

    return output;
}

void FaceTracker::reset()
{
    tracks.clear();
    nextTrackId = 1;
}

float FaceTracker::intersectionOverUnion(const cv::Rect2f &a,
                                         const cv::Rect2f &b)
{
    const cv::Rect2f intersection = a & b;
    const float intersectionArea = std::max(0.0f, intersection.area());
    const float unionArea = a.area() + b.area() - intersectionArea;
    return unionArea > 0.0f ? intersectionArea / unionArea : 0.0f;
}
