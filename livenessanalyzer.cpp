#include "livenessanalyzer.h"

#include <QObject>
#include <algorithm>
#include <cmath>

QString LivenessAnalyzer::update(int trackId, const FaceDetection &face)
{
    State &state = states[trackId];

    const cv::Point2f center(
        face.box.x + face.box.width * 0.5f,
        face.box.y + face.box.height * 0.5f);

    const float width = static_cast<float>(std::max(1, face.box.width));
    const float height = static_cast<float>(std::max(1, face.box.height));

    const cv::Point2f noseOffset(
        (face.landmarks[2].x - center.x) / width,
        (face.landmarks[2].y - center.y) / height);

    const float area = static_cast<float>(std::max(1, face.box.area()));

    if (!state.initialized) {
        state.initialized = true;
        state.firstCenter = center;
        state.firstNoseOffset = noseOffset;
        state.firstArea = area;
    }

    const float normalization =
        static_cast<float>(std::max(1, std::max(face.box.width, face.box.height)));

    const float motion =
        static_cast<float>(cv::norm(center - state.firstCenter)) / normalization;
    const float poseChange =
        static_cast<float>(cv::norm(noseOffset - state.firstNoseOffset));
    const float scaleChange =
        std::abs(area - state.firstArea) / std::max(1.0f, state.firstArea);

    state.maxMotion = std::max(state.maxMotion, motion);
    state.maxPoseChange = std::max(state.maxPoseChange, poseChange);
    state.maxScaleChange = std::max(state.maxScaleChange, scaleChange);
    ++state.frames;

    const bool dynamicPassed =
        state.frames >= 8
        && (state.maxMotion > 0.035f
            || state.maxPoseChange > 0.018f
            || state.maxScaleChange > 0.06f);

    if (dynamicPassed) {
        return QObject::tr("动态通过");
    }
    if (state.frames < 8) {
        return QObject::tr("活体采集中");
    }
    return QObject::tr("请轻微转头");
}

void LivenessAnalyzer::reset()
{
    states.clear();
}
