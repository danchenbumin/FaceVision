#include "facematcher.h"

#include <cmath>
#include <iostream>

namespace {

cv::Mat unitFeature(int position)
{
    cv::Mat feature = cv::Mat::zeros(1, 128, CV_32F);
    feature.at<float>(0, position) = 1.0f;
    return feature;
}

}

int main()
{
    PersonInfo alice;
    alice.id = 1;
    alice.name = "Alice";
    alice.embeddings = {unitFeature(0), unitFeature(1)};

    PersonInfo bob;
    bob.id = 2;
    bob.name = "Bob";
    bob.embeddings = {unitFeature(10)};

    QVector<PersonInfo> persons;
    persons << alice << bob;
    FaceMatcher matcher;
    matcher.rebuild(persons);

    const FaceMatchResult result = matcher.match(unitFeature(10));
    if (result.personId != 2 || result.name != "Bob"
        || std::abs(result.similarity - 1.0f) > 1e-5f) {
        std::cerr << "Face matcher returned an unexpected result.\n";
        return 1;
    }
    if (matcher.personCount() != 2 || matcher.sampleCount() != 3) {
        std::cerr << "Face matcher index statistics are incorrect.\n";
        return 2;
    }

    std::cout << "Face matcher test passed.\n";
    return 0;
}
