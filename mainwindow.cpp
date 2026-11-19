#include "mainwindow.h"
#include "apppaths.h"
#include "ui_mainwindow.h"
#include "facesamplestorage.h"
#include "recognitionlogstorage.h"
#include "personmanagerdialog.h"
#include <QMessageBox>
#include <QtConcurrent/QtConcurrentRun>
#include <QDir>
#include <QFileInfo>
#include <QPainter>
#include <QPen>
#include <QFileDialog>
#include <QDateTime>
#include <QDebug>
#include <QInputDialog>
#include <QCoreApplication>
#include <QEventLoop>
#include <QDialog>
#include <QFrame>
#include <QGraphicsDropShadowEffect>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidgetItem>
#include <QPushButton>
#include <QSizePolicy>
#include <QStatusBar>
#include <QVBoxLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <exception>

namespace {

class CameraTimerResumeGuard
{
public:
    CameraTimerResumeGuard(QTimer *timer, bool resume)
        : timer(timer), resume(resume)
    {
        if (resume && timer) {
            timer->stop();
        }
    }

    ~CameraTimerResumeGuard()
    {
        if (resume && timer) {
            timer->start(45);
        }
    }

private:
    QTimer *timer = nullptr;
    bool resume = false;
};

}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setupModernUi();

    timer = new QTimer(this);
    timer->setTimerType(Qt::PreciseTimer);
    frameWatcher = new QFutureWatcher<FrameProcessingResult>(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);
    connect(frameWatcher, &QFutureWatcher<FrameProcessingResult>::finished,
            this, &MainWindow::handleProcessedFrame);

    connect(ui->btnStartStop, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(ui->btnLoadImage, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(ui->btnRegister, &QPushButton::clicked, this, &MainWindow::onRegister);
    connect(ui->btnManageDB, &QPushButton::clicked, this, &MainWindow::onManageDB);
    connect(ui->btnClearLogs, &QPushButton::clicked, this, &MainWindow::onClearLogs);
    connect(ui->logList, &QListWidget::itemDoubleClicked,
            this, &MainWindow::onLogDoubleClicked);

    systemReady = initializeSystem();
    setRecognitionControlsEnabled(systemReady);
}

MainWindow::~MainWindow()
{
    if (timer) {
        timer->stop();
    }
    waitForFrameProcessing();
    if (cap.isOpened()) {
        cap.release();
    }
    delete recognizer;
    delete ui;
}

bool MainWindow::initializeSystem()
{
    QString pathError;
    if (!AppPaths::ensureDataDirectories(&pathError)) {
        QMessageBox::critical(this, tr("初始化失败"), pathError);
        statusBar()->showMessage(tr("数据目录初始化失败"));
        return false;
    }

    const QString modelDir = AppPaths::modelDirectory();
    const QString detectionModel = QDir(modelDir).filePath(
        QStringLiteral("face_detection_yunet_2023mar.onnx"));
    const QString recognitionModel = QDir(modelDir).filePath(
        QStringLiteral("face_recognition_sface_2021dec.onnx"));

    if (!QFileInfo::exists(detectionModel) || !QFileInfo::exists(recognitionModel)) {
        QMessageBox::critical(
            this, tr("模型文件缺失"),
            tr("没有在以下目录找到完整模型文件：\n%1\n\n"
               "请确认 models 文件夹与程序位于同一目录。")
                .arg(modelDir));
        statusBar()->showMessage(tr("模型文件缺失"));
        return false;
    }

    try {
        recognizer = new FaceRecognizer(detectionModel.toStdString(),
                                        recognitionModel.toStdString());
    } catch (const cv::Exception &error) {
        QMessageBox::critical(this, tr("OpenCV 模型加载失败"),
                              QString::fromLocal8Bit(error.what()));
        statusBar()->showMessage(tr("模型加载失败"));
        return false;
    } catch (const std::exception &error) {
        QMessageBox::critical(this, tr("模型加载失败"),
                              QString::fromLocal8Bit(error.what()));
        statusBar()->showMessage(tr("模型加载失败"));
        return false;
    }

    database = new FaceDatabase(this);
    if (!database->init(AppPaths::databaseFile())) {
        QMessageBox::critical(this, tr("数据库错误"), database->lastError());
        statusBar()->showMessage(tr("数据库初始化失败"));
        return false;
    }

    refreshPersonCache();
    loadRecognitionLogs();
    statusBar()->showMessage(
        tr("系统就绪 · %1人 / %2个样本 · %3")
            .arg(matcher.personCount())
            .arg(matcher.sampleCount())
            .arg(matcher.searchModeName()));
    return true;
}

void MainWindow::setRecognitionControlsEnabled(bool enabled)
{
    ui->btnStartStop->setEnabled(enabled);
    ui->btnLoadImage->setEnabled(enabled);
    ui->btnRegister->setEnabled(enabled);
    ui->btnManageDB->setEnabled(enabled);
    if (QLabel *stateLabel = findChild<QLabel *>(QStringLiteral("systemState"))) {
        stateLabel->setText(enabled ? tr("●  系统就绪")
                                    : tr("●  初始化失败"));
    }
}

void MainWindow::onStartStop()
{
    if (!isCameraRunning) {
        waitForFrameProcessing();
        cap.open(0);
        if (!cap.isOpened()) {
            QMessageBox::critical(this, tr("摄像头错误"), tr("无法打开摄像头，请检查设备占用和权限。"));
            return;
        }
        isImageMode = false;
        tracker.reset();
        livenessAnalyzer.reset();
        timer->start(45);
        isCameraRunning = true;
        ui->btnStartStop->setText(tr("停止摄像头"));
        statusBar()->showMessage(tr("摄像头运行中"));
    } else {
        timer->stop();
        cap.release();
        isCameraRunning = false;
        waitForFrameProcessing();
        identitiesInPreviousFrame.clear();
        tracker.reset();
        livenessAnalyzer.reset();
        ui->btnStartStop->setText(tr("启动摄像头"));
        ui->videoLabel->setText(tr("等待摄像头或图片输入"));
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        statusBar()->showMessage(tr("摄像头已停止"), 3000);
    }
}

void MainWindow::onLoadImage()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, tr("选择图片"), QString(),
        tr("图片 (*.png *.jpg *.jpeg *.bmp *.webp)"));
    if (fileName.isEmpty()) return;
    QString imageError;
    cv::Mat img = FaceSampleStorage::loadImage(fileName, &imageError);
    if (img.empty()) {
        QMessageBox::warning(this, tr("图片错误"), imageError);
        return;
    }

    // 静态图片和摄像头共用同一个显示区域。载入图片时先停止摄像头，
    // 防止下一次定时器触发后摄像头画面立刻覆盖刚载入的图片。
    if (isCameraRunning) {
        timer->stop();
        cap.release();
        isCameraRunning = false;
        ui->btnStartStop->setText(tr("启动摄像头"));
    }
    waitForFrameProcessing();
    currentFrame = img.clone();
    identitiesInPreviousFrame.clear();
    tracker.reset();
    livenessAnalyzer.reset();
    isImageMode = true;
    processFrame();
    statusBar()->showMessage(tr("已加载图片：%1").arg(fileName), 5000);
}

void MainWindow::setupModernUi()
{
    setWindowTitle(tr("FaceVision · 智能人脸识别系统"));
    resize(1180, 760);
    setMinimumSize(1020, 680);
    ui->menubar->hide();

    auto addShadow = [](QWidget *widget, int blurRadius = 30) {
        auto *shadow = new QGraphicsDropShadowEffect(widget);
        shadow->setBlurRadius(blurRadius);
        shadow->setOffset(0, 7);
        shadow->setColor(QColor(15, 23, 42, 28));
        widget->setGraphicsEffect(shadow);
    };

    auto *rootLayout = new QVBoxLayout(ui->centralwidget);
    rootLayout->setContentsMargins(28, 22, 28, 20);
    rootLayout->setSpacing(18);

    auto *headerLayout = new QHBoxLayout;
    auto *titleLayout = new QVBoxLayout;
    titleLayout->setSpacing(2);
    ui->label->setText(tr("FaceVision"));
    ui->label->setObjectName("pageTitle");
    auto *subtitle = new QLabel(
        tr("智能人脸识别 · 人员管理 · 识别记录"), ui->centralwidget);
    subtitle->setObjectName("pageSubtitle");
    titleLayout->addWidget(ui->label);
    titleLayout->addWidget(subtitle);
    headerLayout->addLayout(titleLayout);
    headerLayout->addStretch();
    auto *systemState = new QLabel(tr("●  SYSTEM READY"), ui->centralwidget);
    systemState->setObjectName("systemState");
    headerLayout->addWidget(systemState, 0, Qt::AlignVCenter);
    rootLayout->addLayout(headerLayout);

    auto *contentLayout = new QHBoxLayout;
    contentLayout->setSpacing(18);

    auto *videoCard = new QFrame(ui->centralwidget);
    videoCard->setProperty("card", true);
    auto *videoLayout = new QVBoxLayout(videoCard);
    videoLayout->setContentsMargins(18, 16, 18, 18);
    videoLayout->setSpacing(12);
    auto *previewTitle = new QLabel(tr("实时预览"), videoCard);
    previewTitle->setProperty("sectionTitle", true);
    auto *previewHint = new QLabel(
        tr("支持摄像头实时识别与本地静态图片分析"), videoCard);
    previewHint->setProperty("sectionHint", true);
    videoLayout->addWidget(previewTitle);
    videoLayout->addWidget(previewHint);
    ui->videoLabel->setObjectName("videoLabel");
    ui->videoLabel->setText(tr("等待摄像头或图片输入"));
    ui->videoLabel->setAlignment(Qt::AlignCenter);
    ui->videoLabel->setMinimumSize(500, 430);
    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
    videoLayout->addWidget(ui->videoLabel, 1);
    addShadow(videoCard);
    contentLayout->addWidget(videoCard, 6);

    auto *actionCard = new QFrame(ui->centralwidget);
    actionCard->setProperty("card", true);
    actionCard->setMinimumWidth(190);
    actionCard->setMaximumWidth(220);
    auto *actionLayout = new QVBoxLayout(actionCard);
    actionLayout->setContentsMargins(16, 18, 16, 18);
    actionLayout->setSpacing(12);
    auto *actionTitle = new QLabel(tr("快捷操作"), actionCard);
    actionTitle->setProperty("sectionTitle", true);
    auto *actionHint = new QLabel(tr("选择一种输入或管理数据"), actionCard);
    actionHint->setProperty("sectionHint", true);
    actionHint->setWordWrap(true);
    actionLayout->addWidget(actionTitle);
    actionLayout->addWidget(actionHint);
    actionLayout->addSpacing(12);
    ui->btnStartStop->setText(tr("启动摄像头"));
    ui->btnStartStop->setProperty("primary", true);
    ui->btnLoadImage->setText(tr("导入图片"));
    ui->btnRegister->setText(tr("注册人脸"));
    ui->btnManageDB->setText(tr("人员管理"));
    for (QPushButton *button : {ui->btnStartStop, ui->btnLoadImage,
                                ui->btnRegister, ui->btnManageDB}) {
        button->setMinimumHeight(52);
        button->setCursor(Qt::PointingHandCursor);
        actionLayout->addWidget(button);
    }
    actionLayout->addStretch();
    auto *tip = new QLabel(
        tr("提示：载入图片后可直接注册，双击记录可查看抓拍。"), actionCard);
    tip->setObjectName("actionTip");
    tip->setWordWrap(true);
    actionLayout->addWidget(tip);
    addShadow(actionCard);
    contentLayout->addWidget(actionCard, 2);

    auto *logCard = new QFrame(ui->centralwidget);
    logCard->setProperty("card", true);
    logCard->setMinimumWidth(250);
    logCard->setMaximumWidth(300);
    auto *logLayout = new QVBoxLayout(logCard);
    logLayout->setContentsMargins(16, 18, 16, 18);
    logLayout->setSpacing(12);
    ui->label_2->setText(tr("识别记录"));
    ui->label_2->setProperty("sectionTitle", true);
    auto *logHint = new QLabel(tr("最近 200 条识别事件"), logCard);
    logHint->setProperty("sectionHint", true);
    logLayout->addWidget(ui->label_2);
    logLayout->addWidget(logHint);
    ui->logList->setSpacing(5);
    ui->logList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    logLayout->addWidget(ui->logList, 1);
    ui->btnClearLogs->setText(tr("清空记录"));
    ui->btnClearLogs->setProperty("danger", true);
    ui->btnClearLogs->setMinimumHeight(42);
    ui->btnClearLogs->setCursor(Qt::PointingHandCursor);
    logLayout->addWidget(ui->btnClearLogs);
    addShadow(logCard);
    contentLayout->addWidget(logCard, 3);

    rootLayout->addLayout(contentLayout, 1);

    setStyleSheet(R"(
        QWidget {
            font-family: "Microsoft YaHei UI";
        }
        QMainWindow, QWidget#centralwidget {
            background: #F3F6FA;
            color: #172033;
            font-size: 13px;
        }
        QLabel#pageTitle {
            color: #111827;
            font-size: 27px;
            font-weight: 700;
        }
        QLabel#pageSubtitle, QLabel[sectionHint="true"] {
            color: #7A8496;
            font-size: 12px;
        }
        QLabel#systemState {
            color: #168A66;
            background: #E5F7F0;
            border: 1px solid #C9EDDF;
            border-radius: 15px;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 600;
        }
        QFrame[card="true"] {
            background: #FFFFFF;
            border: 1px solid #E7EBF1;
            border-radius: 18px;
        }
        QLabel[sectionTitle="true"] {
            color: #20293A;
            font-size: 16px;
            font-weight: 650;
        }
        QLabel#videoLabel {
            color: #8E9AAF;
            background: #101827;
            border: 2px solid #263449;
            border-radius: 14px;
            font-size: 15px;
        }
        QPushButton {
            color: #344054;
            background: #F8FAFC;
            border: 1px solid #DFE5EC;
            border-radius: 12px;
            padding: 0 16px;
            font-weight: 600;
            text-align: left;
        }
        QPushButton:hover {
            color: #1D4ED8;
            background: #EEF4FF;
            border: 1px solid #AFC8FF;
        }
        QPushButton:pressed {
            color: #1E40AF;
            background: #DCE8FF;
            border-color: #7DA5F5;
        }
        QPushButton[primary="true"] {
            color: white;
            background: #1E3A5F;
            border: 1px solid #1E3A5F;
        }
        QPushButton[primary="true"]:hover {
            background: #28517F;
            border-color: #3C6B9E;
        }
        QPushButton[primary="true"]:pressed {
            background: #142A46;
        }
        QPushButton[danger="true"] {
            color: #B54747;
            background: #FFF8F7;
            border-color: #F4D8D5;
            text-align: center;
        }
        QPushButton[danger="true"]:hover {
            color: #9E2F2F;
            background: #FDECE9;
            border-color: #EABAB4;
        }
        QListWidget {
            color: #344054;
            background: #F8FAFC;
            border: 1px solid #E4E9F0;
            border-radius: 12px;
            padding: 7px;
            outline: none;
        }
        QListWidget::item {
            background: transparent;
            border-radius: 9px;
            padding: 9px 8px;
        }
        QListWidget::item:hover {
            background: #EDF3FA;
        }
        QListWidget::item:selected {
            color: #163F70;
            background: #DCE9F7;
        }
        QLabel#actionTip {
            color: #667085;
            background: #F4F7FA;
            border-radius: 10px;
            padding: 10px;
            font-size: 11px;
        }
        QScrollBar:vertical {
            width: 8px;
            background: transparent;
            margin: 8px 1px;
        }
        QScrollBar::handle:vertical {
            min-height: 24px;
            background: #CBD3DE;
            border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QStatusBar {
            color: #6B7280;
            background: #F3F6FA;
            border: none;
            padding-left: 24px;
        }
    )");
    statusBar()->showMessage(tr("系统就绪"));
}

void MainWindow::processFrame()
{
    if (!systemReady || !recognizer || !database || frameResultPending) {
        return;
    }

    cv::Mat frame;
    bool fromCamera = false;
    if (isCameraRunning) {
        cap >> frame;
        if (frame.empty()) {
            return;
        }
        currentFrame = frame.clone();
        fromCamera = true;
    } else if (isImageMode) {
        frame = currentFrame.clone();
        isImageMode = false;
    } else {
        return;
    }

    frameResultPending = true;
    frameWatcher->setFuture(QtConcurrent::run(
        [this, frame, fromCamera]() {
            return analyzeFrame(frame, fromCamera);
        }));
}

FrameProcessingResult MainWindow::analyzeFrame(const cv::Mat &frame,
                                                bool fromCamera) const
{
    FrameProcessingResult result;
    result.frame = frame.clone();
    result.fromCamera = fromCamera;

    try {
        const std::vector<FaceDetection> faces = recognizer->detectFaces(frame);
        result.faces.reserve(static_cast<qsizetype>(faces.size()));
        for (const FaceDetection &face : faces) {
            ProcessedFace processed;
            processed.detection = face;
            const cv::Mat feature = recognizer->extractFeature(frame, face);
            if (!feature.empty()) {
                const FaceMatchResult match = matcher.match(feature);
                processed.personId = match.personId;
                processed.identity = match.name;
                processed.similarity = match.similarity;
                if (processed.personId < 0
                    || processed.similarity < RecognitionThreshold) {
                    processed.personId = -1;
                    processed.identity = tr("陌生人");
                }
            } else {
                processed.identity = tr("特征提取失败");
            }
            result.faces.append(processed);
        }
    } catch (const cv::Exception &error) {
        result.error = tr("OpenCV 处理失败：%1")
                           .arg(QString::fromLocal8Bit(error.what()));
    } catch (const std::exception &error) {
        result.error = tr("图像处理失败：%1")
                           .arg(QString::fromLocal8Bit(error.what()));
    }
    return result;
}

void MainWindow::handleProcessedFrame()
{
    frameResultPending = false;
    const FrameProcessingResult result = frameWatcher->result();
    if (!result.error.isEmpty()) {
        statusBar()->showMessage(result.error, 5000);
        return;
    }

    std::vector<FaceDetection> detections;
    detections.reserve(static_cast<std::size_t>(result.faces.size()));
    for (const ProcessedFace &processed : result.faces) {
        detections.push_back(processed.detection);
    }

    const QVector<TrackedFace> trackedFaces = tracker.update(detections);
    QVector<FrameAnnotation> annotations;
    QSet<int> recognizedThisFrame;

    for (int index = 0; index < result.faces.size(); ++index) {
        const ProcessedFace &processed = result.faces[index];
        const FaceDetection &face = processed.detection;
        const cv::Rect safeRect = face.box
            & cv::Rect(0, 0, result.frame.cols, result.frame.rows);
        if (safeRect.empty()) {
            continue;
        }

        FrameAnnotation annotation;
        annotation.box = index < trackedFaces.size()
                             ? trackedFaces[index].smoothedBox
                             : safeRect;

        if (processed.identity == tr("特征提取失败")) {
            annotation.color = QColor(239, 68, 68);
            annotation.text = processed.identity;
            annotations.append(annotation);
            continue;
        }

        const int trackId = index < trackedFaces.size()
                                ? trackedFaces[index].trackId
                                : index + 1;
        const QString liveness = result.fromCamera
                                     ? livenessAnalyzer.update(trackId, face)
                                     : tr("静态图片");
        const bool recognized = processed.personId >= 0;
        annotation.color = recognized ? QColor(22, 163, 74)
                                      : QColor(245, 158, 11);
        annotation.text = tr("%1  相似度 %2  ·  %3")
                              .arg(processed.identity)
                              .arg(processed.similarity, 0, 'f', 3)
                              .arg(liveness);
        annotations.append(annotation);

        if (recognized) {
            recognizedThisFrame.insert(processed.personId);
            const bool justAppeared =
                !identitiesInPreviousFrame.contains(processed.personId);
            if (justAppeared
                && (!result.fromCamera || shouldAppendLog(processed.personId))) {
                QString snapshotError;
                const QString snapshotPath = RecognitionLogStorage::saveSnapshot(
                    result.frame, safeRect, &snapshotError);
                const QString sourceType = result.fromCamera ? "Camera" : "Image";
                int logId = -1;
                if (database->addRecognitionLog(
                        processed.personId, processed.identity,
                        processed.similarity, snapshotPath, sourceType, &logId)) {
                    RecognitionLogInfo log;
                    log.id = logId;
                    log.personId = processed.personId;
                    log.nameSnapshot = processed.identity;
                    log.similarity = processed.similarity;
                    log.recognizedAt = QDateTime::currentDateTime().toString(
                        "yyyy-MM-dd HH:mm:ss");
                    log.snapshotPath = snapshotPath;
                    log.sourceType = sourceType;
                    appendRecognitionLogItem(log);
                } else {
                    RecognitionLogStorage::deleteStoredSnapshot(snapshotPath);
                    qWarning() << "Failed to persist recognition log:"
                               << database->lastError();
                }
            }
        }
    }

    identitiesInPreviousFrame = recognizedThisFrame;
    displayImage(result.frame, annotations);
    statusBar()->showMessage(
        tr("检测到 %1 张人脸 · 已注册 %2 人 / %3 个样本 · %4")
            .arg(result.faces.size())
            .arg(matcher.personCount())
            .arg(matcher.sampleCount())
            .arg(matcher.searchModeName()));
}

void MainWindow::waitForFrameProcessing()
{
    if (frameWatcher && frameWatcher->isRunning()) {
        frameWatcher->waitForFinished();
    }
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

bool MainWindow::shouldAppendLog(int personId)
{
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const qint64 lastTime = lastLogTimes.value(personId, 0);
    if (lastTime > 0 && now - lastTime < LogCooldownMs) {
        return false;
    }

    lastLogTimes.insert(personId, now);
    return true;
}

void MainWindow::loadRecognitionLogs()
{
    ui->logList->clear();
    const QVector<RecognitionLogInfo> logs = database->getRecentRecognitionLogs(200);
    // appendRecognitionLogItem inserts at row 0, so replay oldest to newest.
    for (auto iterator = logs.crbegin(); iterator != logs.crend(); ++iterator) {
        appendRecognitionLogItem(*iterator);
    }
}

void MainWindow::appendRecognitionLogItem(const RecognitionLogInfo &log)
{
    const QString source = log.sourceType == "Camera" ? tr("摄像头") : tr("图片");
    const QString time = log.recognizedAt.size() >= 16
                             ? log.recognizedAt.mid(5, 11)
                             : log.recognizedAt;
    auto *item = new QListWidgetItem(
        QString("%1  %2\n%3  %4")
            .arg(time, log.nameSnapshot,
                 tr("相似度 %1").arg(log.similarity, 0, 'f', 2), source));
    item->setData(Qt::UserRole, log.snapshotPath);
    item->setData(Qt::UserRole + 1, log.recognizedAt);
    item->setToolTip(tr("双击查看识别抓拍。"));
    ui->logList->insertItem(0, item);
    while (ui->logList->count() > 200) {
        delete ui->logList->takeItem(ui->logList->count() - 1);
    }
}

void MainWindow::onLogDoubleClicked(QListWidgetItem *item)
{
    if (!item) {
        return;
    }
    const QString storedPath = item->data(Qt::UserRole).toString();
    const QString absolutePath = RecognitionLogStorage::resolveStoredPath(storedPath);
    QPixmap snapshot(absolutePath);
    if (snapshot.isNull()) {
        QMessageBox::information(this, tr("识别抓拍"),
                                 tr("该记录没有可用的抓拍图片。"));
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(tr("识别抓拍"));
    auto *layout = new QVBoxLayout(&dialog);
    auto *details = new QLabel(item->text(), &dialog);
    auto *imageLabel = new QLabel(&dialog);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setPixmap(snapshot.scaled(640, 480, Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation));
    layout->addWidget(details);
    layout->addWidget(imageLabel);
    dialog.exec();
}

void MainWindow::onClearLogs()
{
    if (ui->logList->count() == 0) {
        return;
    }
    if (QMessageBox::question(
            this, tr("清空识别记录"),
            tr("确定清空全部识别记录及对应抓拍吗？"))
        != QMessageBox::Yes) {
        return;
    }
    if (!database->clearRecognitionLogs()) {
        QMessageBox::warning(this, tr("清空失败"), database->lastError());
        return;
    }
    ui->logList->clear();
    lastLogTimes.clear();
}

void MainWindow::refreshPersonCache()
{
    if (!database) {
        personCache.clear();
        matcher.rebuild(personCache);
        return;
    }
    personCache = database->getAllPersons();
    matcher.rebuild(personCache);
}

void MainWindow::displayImage(const cv::Mat &mat,
                              const QVector<FrameAnnotation> &annotations)
{
    if (mat.empty()) {
        return;
    }

    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage image(rgb.data, rgb.cols, rgb.rows, rgb.step,
                 QImage::Format_RGB888);
    image = image.copy();

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    QFont font(QStringLiteral("Microsoft YaHei UI"));
    font.setPixelSize(std::max(15, image.width() / 70));
    font.setBold(true);
    painter.setFont(font);

    for (const FrameAnnotation &annotation : annotations) {
        const QRect box(annotation.box.x, annotation.box.y,
                        annotation.box.width, annotation.box.height);
        QPen pen(annotation.color, std::max(2, image.width() / 500));
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(box, 7, 7);

        const QFontMetrics metrics(font);
        const int horizontalPadding = 8;
        const int labelHeight = metrics.height() + 8;
        const int labelWidth = std::min(
            image.width() - 4,
            metrics.horizontalAdvance(annotation.text) + horizontalPadding * 2);
        const int labelX = std::clamp(box.left(), 2,
                                      std::max(2, image.width() - labelWidth - 2));
        int labelY = box.top() - labelHeight;
        if (labelY < 2) {
            labelY = std::min(image.height() - labelHeight - 2,
                              box.bottom() + 4);
        }
        const QRect labelRect(labelX, labelY, labelWidth, labelHeight);
        QColor background = annotation.color;
        background.setAlpha(220);
        painter.setPen(Qt::NoPen);
        painter.setBrush(background);
        painter.drawRoundedRect(labelRect, 5, 5);
        painter.setPen(Qt::white);
        painter.drawText(labelRect.adjusted(horizontalPadding, 0,
                                            -horizontalPadding, 0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         annotation.text);
    }
    painter.end();

    ui->videoLabel->setPixmap(
        QPixmap::fromImage(image).scaled(ui->videoLabel->size(),
                                         Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation));
}

void MainWindow::onRegister()
{
    CameraTimerResumeGuard resumeGuard(timer, isCameraRunning);
    waitForFrameProcessing();
    if (currentFrame.empty()) {
        QMessageBox::information(this, tr("提示"), tr("请先启动摄像头或导入一张图片。"));
        return;
    }
    std::vector<FaceDetection> faces = recognizer->detectFaces(currentFrame);
    qDebug() << "Detected faces for registration:" << faces.size();
    for (const auto &face : faces)
    {
        const cv::Rect &rect = face.box;
        qDebug() << "  Rect:" << rect.x << rect.y << rect.width << rect.height
                 << "confidence:" << face.confidence;
    }

    if (faces.empty()) {
        QMessageBox::information(this, tr("未检测到人脸"), tr("当前画面中没有检测到人脸，请调整光线、距离或角度。"));
        return;
    }
    if (faces.size() > 1) {
        QMessageBox::information(this, tr("人脸数量不符合要求"), tr("注册时画面中必须恰好有一张人脸。"));
        return;
    }
    const cv::Rect faceRect = faces[0].box
                              & cv::Rect(0,0,currentFrame.cols,currentFrame.rows);
    if (faceRect.empty()) {
        QMessageBox::warning(this, tr("人脸区域无效"), tr("检测到的人脸区域无效，请重新采集。"));
        return;
    }
    cv::Mat alignedFace;
    cv::Mat feature = recognizer->extractFeature(currentFrame, faces[0],
                                                 &alignedFace);
    if (feature.empty()) {
        QMessageBox::warning(this, tr("特征提取失败"), tr("人脸对齐或特征提取失败，请更换清晰、正面的图像。"));
        return;
    }

    bool ok = false;
    const QString personCode = QInputDialog::getText(
        this, tr("注册人脸"), tr("请输入人员编号："),
        QLineEdit::Normal, QString(), &ok).trimmed();
    if (!ok || personCode.isEmpty()) {
        return;
    }

    PersonInfo existingPerson;
    const bool personExists = database->findPersonByCode(personCode, existingPerson);
    if (!personExists && !database->lastError().isEmpty()) {
        QMessageBox::warning(this, tr("查询失败"), database->lastError());
        return;
    }

    QString name;
    QString department;
    if (personExists) {
        const auto answer = QMessageBox::question(
            this,
            tr("追加人脸样本"),
            tr("编号 %1 已属于“%2”。是否把当前人脸作为该人员的新样本？")
                .arg(personCode, existingPerson.name));
        if (answer != QMessageBox::Yes) {
            return;
        }
        name = existingPerson.name;
        department = existingPerson.department;
    } else {
        name = QInputDialog::getText(
            this, tr("注册人脸"), tr("请输入姓名："),
            QLineEdit::Normal, QString(), &ok).trimmed();
        if (!ok || name.isEmpty()) {
            return;
        }
        department = QInputDialog::getText(
            this, tr("注册人脸"), tr("请输入部门（可留空）："),
            QLineEdit::Normal, QString(), &ok).trimmed();
        if (!ok) {
            return;
        }
    }

    QString storageError;
    const QString sampleImagePath = FaceSampleStorage::saveAlignedFace(
        alignedFace, &storageError);
    if (sampleImagePath.isEmpty()) {
        QMessageBox::warning(this, tr("保存样本失败"), storageError);
        return;
    }

    bool createdPerson = false;
    if (!database->registerFaceSample(personCode, name, department, feature,
                                      sampleImagePath, nullptr, &createdPerson)) {
        FaceSampleStorage::deleteStoredImage(sampleImagePath);
        QMessageBox::warning(this, tr("注册失败"), database->lastError());
        return;
    }

    refreshPersonCache();
    const PersonInfo updatedPerson = [&]() {
        for (const PersonInfo &person : personCache) {
            if (person.personCode == personCode) {
                return person;
            }
        }
        return PersonInfo{};
    }();
    QMessageBox::information(
        this,
        tr("注册成功"),
        createdPerson
            ? tr("人员“%1”注册成功，当前有1个人脸样本。").arg(name)
            : tr("已为“%1”追加人脸样本，当前共有%2个样本。")
                  .arg(name)
                  .arg(updatedPerson.embeddings.size()));
}

void MainWindow::onManageDB()
{
    CameraTimerResumeGuard resumeGuard(timer, isCameraRunning);
    waitForFrameProcessing();
    PersonManagerDialog dialog(database, recognizer, this);
    dialog.exec();
    refreshPersonCache();
}
