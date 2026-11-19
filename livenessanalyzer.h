#ifndef LIVENESSANALYZER_H
#define LIVENESSANALYZER_H

#include "facerecognizer.h"

#include <QHash>
#include <QString>

class LivenessAnalyzer
{
public:
    QString update(int trackId, const FaceDetection &face);
    void reset();

private:
    struct State {
        bool initialized = false;
        cv::Point2f firstCenter;
        cv::Point2f firstNoseOffset;
        float firstArea = 0.0f;
        float maxMotion = 0.0f;
        float maxPoseChange = 0.0f;
        float maxScaleChange = 0.0f;
        int frames = 0;
    };

    QHash<int, State> states;
};

#endif // LIVENESSANALYZER_H
