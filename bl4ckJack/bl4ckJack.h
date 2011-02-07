#ifndef BL4CKJACK_H
#define BL4CKJACK_H

#ifndef VERSION
#define VERSION "0.42-3"
#endif

#define DEFAULT_PORT		40201

#include <QtGui>
#include <QtGui/QMainWindow>
#include <QSystemTrayIcon>
#include <QDialog>

#include "ui_bl4ckjack.h"
#include "bl4ckJack_config.h"
#include "bl4ckJack_module.h"
#include "bl4ckJack_timer.h"

// distributed coding
#include "bl4ckJack_distributed.h"
#include "bl4ckJack_tableview.h"

class QAction;
class QMenu;

extern QSettings *settings;

typedef struct {
	bl4ckJack_Module *moduleInfo;

	fbl4ckJackInit pfbl4ckJackInit;
	fbl4ckJackMatch pfbl4ckJackMatch;
	fbl4ckJackInfo pfbl4ckJackInfo;
	fbl4ckJackFree pfbl4ckJackFree;
	fbl4ckJackGenerate pfbl4ckJackGenerate;
} bl4ckJackModuleList;

extern QList<bl4ckJackModuleList *> bl4ckJackModules;

class bl4ckJack : public QMainWindow
{
	Q_OBJECT

public:
	
	Ui::bl4ckJackClass ui;
	bl4ckJack_Module **moduleList;
	TableModel *tblHashView;

	bl4ckJack(QWidget *parent = 0, Qt::WFlags flags = 0);
	~bl4ckJack();
	
    void setVisible(bool visible);
	void writeSettings();

public slots:
	void updateUIFileAdd(QString, QString, QString, float);
	void updateBruteStatus(int, int, QString);
	void updateBruteLabels(double, QString, qint64);
	void updateBrutePassword(QString, QString);

 protected:
    void closeEvent(QCloseEvent *event);

 private slots:
    void iconActivated(QSystemTrayIcon::ActivationReason reason);
    void messageClicked(void);
	void showProperties(void);
	
	//! Control functions
	void start(void);
	void stop(void);
	void pause(void);

	void hashTableInputHash(void);
	void hashTableInputFile(void);
	void hashTableDeleteHash(void);
	void hashTableClear(void);

	void PasswordSaveFileTableClear(void);
	void PasswordSaveFileTable(void);
	

 private:
	
	int moduleCount;

	void moduleProcessDir(QString);

	void createMenuActions();

	//! System Tray

	bool isVis;
	void createSystemTrayActions();
    void createTrayIcon();
	void createUIMenus();
	void createStatusBar();

    QAction *minimizeAction;
    QAction *restoreAction;
    QAction *quitAction;
	
	QIcon	bl4ckJackIcon;
	QSystemTrayIcon *trayIcon;
	QMenu *trayIconMenu;
	
	//! Right Click Menu
	QAction *tblHashAddHash;
	QAction *tblHashAddFile;
	QAction *tblHashDelEntry;
	QAction *tblHashClear;
	//QAction *tblHashEditEntry;

	QAction *tblPasswordClear;
	QAction *tblPasswordSaveFile;

	QMenu *tblHashMenu;
	QMenu *tblPasswordMenu;

	bl4ckJackBrute *bruteThread;


};

extern bl4ckJack *bJMain;


class InputHashWorker : public QThread {
	 Q_OBJECT

public:

	InputHashWorker(QObject *parent) : QThread(parent) { 
	}
		
	void setFile(QString filename) {
		this->filename = filename;
	}

	void msleep(unsigned long x) {
		return QThread::msleep(x);
	}

	qint64 getTotal() {
		return this->total;
	}

	qint64 getCurrent() {
		return this->current;
	}

    virtual void run() 
    { 
		
		/*
		QTimer::singleShot(0, this, SLOT(doWork()));
		exec();
		*/
		doWork();
    }

signals:
	void updateUIFileAdd(QString, QString, QString, float);


 private:
	 /* should probably lock around current/total */
	 qint64 current;
	 qint64 total;
	 QString filename;

public slots:

    void doWork()
    {

		float status = 0;
		Timer *t = new Timer;
		QString qBuf, qVersion, qName;
		qint64 hashCount=0;
        // qDebug() << "executing thread id - " << QThread::currentThreadId();
		
		QFile *file = new QFile(this->filename); //(this->filename);
		// open file contents, put contents into textbox
		if (file->open(QFile::ReadWrite)) {
			 
			char *buf=NULL;

			qint64 lineLength = 0;
			qint64 fileSize = this->total = file->size();
			uchar *memory = file->map(0, fileSize);

			buf = (char *) memory;
			
			qint64 start = t->StartTiming();
			while(buf) {
			
				char *newLine = NULL;
				if((newLine=strchr(buf, '\n')))
					*newLine = '\0';
				if(newLine && (*(newLine - 1) == '\r'))
					*(newLine - 1) = '\0';

				this->current = (unsigned long long) buf - (unsigned long long) memory;

				bl4ckJackModuleList *s = NULL;
				
				hashCount++;
				//qDebug() << "Timecheck: " << t->ElapsedTiming(start, t->StopTiming());
				if( (hashCount % 5 == 0) && (t->ElapsedTiming(start, t->StopTiming()) >= 1000) ) {
					qDebug() << "Processing " << (hashCount) << " hash/sec";
					start = t->StartTiming();
					hashCount = 0;
				}

				foreach( s, bl4ckJackModules ) {
					if(s->pfbl4ckJackMatch(buf) == true) {
						qVersion.sprintf("%.2f", s->moduleInfo->version);
						qName = s->moduleInfo->name;
						qBuf = buf;
						// qDebug() << "Matched against " << s->moduleInfo->name << " with " << buf << " successfully.";
						status = (this->current + 1);
						status = status / this->total;
						status = status * 100.0;
						emit updateUIFileAdd(qName, qVersion, qBuf, status);
						break;
					}
				}

				if(newLine) {
					buf = newLine + 1;
					if((*(newLine - 1) == '\0'))
						*(newLine - 1) = '\r';
					*newLine = '\n';
					newLine = NULL;
				} else {
					break /*buf = NULL */;
				}
			}

			file->unmap(memory); 
			file->close();

		}
		delete t;
		delete file;
		// this->quit();
    }

};

#endif // BL4CKJACK_H
