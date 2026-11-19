#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QColor>
#include <QFutureWatcher>
#include <QHash>
#include <QMainWindow>
#include <QSet>
#include <QTimer>
#include <QVector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "facedatabase.h"
#include "facematcher.h"
#include "facerecognizer.h"
#include "facetracker.h"
#include "livenessanalyzer.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class QListWidgetItem;

struct FrameAnnotation
{
    cv::Rect box;
    QString text;
    QColor color;
};

struct ProcessedFace
{
    FaceDetection detection;
    int personId = -1;
    QString identity;
    float similarity = 0.0f;
};

struct FrameProcessingResult
{
    cv::Mat frame;
    QVector<ProcessedFace> faces;
    QString error;
    bool fromCamera = false;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void onStartStop();
    void onLoadImage();
    void onRegister();
    void onManageDB();
    void onClearLogs();
    void onLogDoubleClicked(QListWidgetItem *item);
    void processFrame();
    void handleProcessedFrame();

private:
    bool initializeSystem();
    FrameProcessingResult analyzeFrame(const cv::Mat &frame,
                                       bool fromCamera) const;
    void waitForFrameProcessing();
    void setRecognitionControlsEnabled(bool enabled);
    void displayImage(const cv::Mat &mat,
                      const QVector<FrameAnnotation> &annotations = {});
    bool shouldAppendLog(int personId);
    void refreshPersonCache();
    void loadRecognitionLogs();
    void appendRecognitionLogItem(const RecognitionLogInfo &log);
    void setupModernUi();

    Ui::MainWindow *ui = nullptr;
    QTimer *timer = nullptr;
    QFutureWatcher<FrameProcessingResult> *frameWatcher = nullptr;
    cv::VideoCapture cap;
    cv::Mat currentFrame;
    bool isCameraRunning = false;
    bool isImageMode = false;
    bool systemReady = false;
    bool frameResultPending = false;
    FaceRecognizer *recognizer = nullptr;
    FaceDatabase *database = nullptr;
    QVector<PersonInfo> personCache;
    FaceMatcher matcher;
    FaceTracker tracker;
    LivenessAnalyzer livenessAnalyzer;
    QHash<int, qint64> lastLogTimes;
    QSet<int> identitiesInPreviousFrame;

    static constexpr float RecognitionThreshold = 0.363f;
    static constexpr qint64 LogCooldownMs = 5000;
};

#endif // MAINWINDOW_H
