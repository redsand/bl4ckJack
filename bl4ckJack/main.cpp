#include "bl4ckjack.h"
#include <QtGui/QApplication>

bl4ckJack *bJMain=NULL;

int main(int argc, char *argv[])
{
	int ret = 0;
	QApplication a(argc, argv);

	if (!QSystemTrayIcon::isSystemTrayAvailable()) {
         QMessageBox::critical(0, QObject::tr("Systray"),
                               QObject::tr("System tray not found on system."));
         return 1;
    }
    QApplication::setQuitOnLastWindowClosed(false);

	bJMain = new bl4ckJack;
	bJMain->show();
	ret = a.exec();
	bJMain->writeSettings();
	return ret;
}
