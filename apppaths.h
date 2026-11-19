#ifndef APPPATHS_H
#define APPPATHS_H

#include <QString>

namespace AppPaths {

QString applicationDirectory();
QString dataDirectory();
QString databaseFile();
QString modelDirectory();
QString faceSampleDirectory();
QString recognitionSnapshotDirectory();
bool ensureDataDirectories(QString *errorMessage = nullptr);

}

#endif // APPPATHS_H
