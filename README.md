# FaceVision

2026 年 7 月，四川大学计算机科学与技术工程实训 Qt 开发项目。

一个运行在 Windows 上的桌面人脸识别程序。界面用 Qt 6 Widgets 写，人脸检测和识别调用 OpenCV 的 YuNet、SFace 模型，人员信息、人脸样本和识别记录保存在本地 SQLite 数据库中。程序可以接摄像头做实时识别，也可以导入本地图片分析，注册过的人会在画面中直接标出姓名和相似度。

## 功能

- 摄像头实时识别，也支持导入 PNG、JPG、BMP、WebP 图片，路径带中文可以正常读取
- 同屏多人脸检测：YuNet 按不同尺寸和阈值跑三轮，再用 IoU 合并重复框
- 注册时要求画面中只有一张脸；同一编号再次注册会追加为新的样本，一个人可以存多张不同角度和表情的照片
- 人员管理窗口支持增删改查、按编号或姓名搜索、批量导入样本、查看缩略图、删除单张样本
- 识别成功后写入日志（时间、姓名、相似度、来源）并保存人脸抓拍，双击日志可以查看图片
- 摄像头模式下同一人只记录一次，离开超过 5 秒后再次出现才会重新记录
- 画面中的人脸框按 IoU 关联相邻帧并做指数平滑，减少抖动
- 摄像头模式下根据人脸位置、尺度和关键点变化给出活体提示

## 结构与流程

程序分三层：

- 界面层：`mainwindow` 负责摄像头、图片导入、注册、日志和画面绘制，`personmanagerdialog` 负责人员和样本管理。
- 算法层：`facerecognizer` 做检测、对齐和特征提取，`facematcher` 做相似度检索，`facetracker` 平滑人脸框，`livenessanalyzer` 给出活体提示。
- 数据层：`facedatabase` 管 SQLite 三张表（people、face_samples、recognition_logs），`facesamplestorage` 和 `recognitionlogstorage` 管样本图和抓拍文件，`apppaths` 决定模型和数据目录。

摄像头画面由 QTimer 每 45 ms 取一帧，导入的图片直接读进内存，两者走同一套流程：取帧 → 交给工作线程 → YuNet 按 1280、640 两种缩放尺寸和 0.90、0.85 两个阈值组合跑三轮检测，再按 IoU 合并重复框 → SFace 根据五个关键点对齐，提取 128 维特征并做 L2 归一化 → 和库中所有样本、每个人计算出的中心特征比余弦相似度 → 最高分不低于 0.363 判为已注册人员，否则显示“陌生人” → 回到界面画框，符合条件时写日志并保存抓拍。

样本少于 200 条时，所有特征拼成一个矩阵，一次矩阵乘法算完全部相似度；超过 200 条后改用 FLANN KD 树查询 8 个近邻，再按 cos = 1 - d²/2 换算成余弦值。

检测、特征提取和匹配都在 QtConcurrent 工作线程里执行，上一帧没处理完就丢弃新帧，界面不会卡住。视频模式下人脸框用 IoU 关联相邻帧并做指数平滑，同时根据位置、尺度和关键点变化给出活体提示。

## 编译

需要：

- Windows 10/11 64 位
- Qt 6.5 以上，MSVC 2022 64 位
- OpenCV 4.5 以上，MSVC 64 位
- CMake 3.21 以上

Qt 和 OpenCV 必须使用同一套编译器和同样位数，MSVC 和 MinGW 不能混用。

用 Qt Creator 打开 `CMakeLists.txt`，选择 Qt 6 的 MSVC 2022 64 位 Kit，在 CMake 配置里把 `OpenCV_DIR` 指向 `OpenCVConfig.cmake` 所在目录，然后构建运行。

命令行：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="<Qt msvc2022_64 目录>" `
  -DOpenCV_DIR="<OpenCVConfig.cmake 所在目录>"
cmake --build build --parallel
```

构建完成后会自动复制模型和 OpenCV 运行库，并调用 windeployqt 部署 Qt 插件和 SQLite 驱动。

## 运行

`models` 目录要和 exe 放在同一层，里面是两个 ONNX 模型，仓库里已经带上。`faces.db`、样本图片和识别抓拍默认写在 exe 旁的 `data` 目录，该目录不可写时会改用系统的应用数据目录。

## 目录

```text
FaceVision/
├─ CMakeLists.txt           CMake 工程
├─ CMakePresets.json        构建预设
├─ main.cpp
├─ mainwindow.cpp/.h/.ui    主界面，摄像头与图片流程，日志和结果绘制
├─ facerecognizer.cpp/.h    YuNet 多轮检测、NMS、SFace 对齐与特征提取
├─ facematcher.cpp/.h       样本矩阵、人员中心特征、KD 树检索
├─ facetracker.cpp/.h       IoU 轨迹关联与框平滑
├─ livenessanalyzer.cpp/.h  动态活体提示
├─ facedatabase.cpp/.h      SQLite 建表与人员、样本、日志读写
├─ personmanagerdialog.*    人员管理窗口
├─ facesamplestorage.*      样本图片存取
├─ recognitionlogstorage.*  识别抓拍存取
├─ apppaths.*               模型与数据目录
├─ imageutils.*             图像编解码
├─ models/                  YuNet、SFace 模型
├─ tests/                   数据库、匹配器和检测冒烟测试
└─ docs/                    项目验收说明
```

## 说明

- 活体模块只是根据连续帧的运动和尺度变化做提示，不是安全级防伪，不能用于门禁或金融场景
- 识别阈值固定为 0.363，没有做界面标定，换摄像头或换数据集需要重新验证
- 仓库不包含任何真实人脸数据、数据库和抓拍图片
- 人脸属于敏感生物信息，实际使用时需要取得对方同意
