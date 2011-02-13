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

//! Global Settings Manager
extern QSettings *settings; /**< global settings manager. */

//! bl4ckJackModuleList A set of exported module functions (structure)
typedef struct {
	bl4ckJack_Module *moduleInfo; /**< moduleInfo structure */

	fbl4ckJackInit pfbl4ckJackInit; /**< CPU init bruteforce */
	fbl4ckJackMatch pfbl4ckJackMatch; /**< CPU match bruteforce */
	fbl4ckJackInfo pfbl4ckJackInfo; /**< moduleInfo */
	fbl4ckJackFree pfbl4ckJackFree; /**< /CPU free bruteforce */
	fbl4ckJackGenerate pfbl4ckJackGenerate; /**< CPU bruteforce generate hash */

	/* GPU */
	fbl4ckJackInitGPU pfbl4ckJackInitGPU; /**< GPU bruteforce generate hash */
	fbl4ckJackFreeGPU pfbl4ckJackFreeGPU; /**< GPU free bruteforce */
	fbl4ckJackGenerateGPU pfbl4ckJackGenerateGPU; /**< GPU bruteforce generate hash */
	
//! bl4ckJackModuleList A set of exported module functions (structure)
} bl4ckJackModuleList;

//! List of exported module functions (structure)
extern QList<bl4ckJackModuleList *> bl4ckJackModules; /**< exported list of available CPU/GPU modules */

//! Main class controls and interacts with the QT GUI
/**
 * Main class controls and interacts with the QT GUI
 */
 
class bl4ckJack : public QMainWindow
{
	Q_OBJECT

public:
	
	Ui::bl4ckJackClass ui; /**< QT generated user interface */
	bl4ckJack_Module **moduleList; /**< list of available CPU/GPU modules */
	TableModel *tblHashView; /**< QT Hash/Salt Table View */

	//! QT GUI constructor
	/**
	  * QT GUI constructor
	  * Responsible for interacting with the GUI main thread and all sub-threads for processing.
	  * @param parent widget pointer
	  * @param Qt::WFlags
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	bl4ckJack(QWidget *parent = 0, Qt::WFlags flags = 0); 
	
	/**
	  * QT GUI deconstructor
	  * Responsible for freeing our allocated objects.
      * @see bl4ckJack()
      * @see bl4ckJack()
      * @return None
	  */
	~bl4ckJack(); /**< QT GUI destructor */
	
	//! Show/Hide QT GUI
	/**
	  * Show/Hide QT GUI
	  * Responsible for showing or hiding the QT GUI
	  * @param bool visible
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
    void setVisible(bool visible);
	
	//! Save QT Settings
	/**
	  * Save QT Settings
	  * Responsible for saving our settings configured by user interaction.
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void writeSettings();

public slots:

	//! Update QT Gui with new hashes for processing
	/**
	  * Update QT Gui with new hashes for processing
	  * Responsible for adding new hashes to the GUI.
	  * @param QString
	  * @param QString
	  * @param QString
	  * @param float
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void updateUIFileAdd(QString, QString, QString, float);
	
	//! Update QT Gui with current bruteforcing status
	/**
	  * Update QT Gui with current bruteforcing status
	  * Responsible for updating QT Gui thread with bruteforcing status including percentage of completion, and update notices.
	  * @param int priority
	  * @param int percentage completion
	  * @param QString update status
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void updateBruteStatus(int, int, QString);
	
	//! Update QT Gui Bruteforcing Labels
	/**
	  * Update QT Gui Bruteforcing Labels
	  * Responsible for updating QT Gui thread with bruteforcing labels including passwords/sec, completion time, etc.
	  * @param double passwords per second
	  * @param QString completion time
	  * @param qint64 cracked passwords
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void updateBruteLabels(double, QString, qint64);
	
	//! Update QT Gui with any saved passwords during bruteforcing process
	/**
	  * Update QT Gui with any saved passwords during bruteforcing process
	  * Responsible for updating QT Gui thread with any saved passwords during bruteforcing process.
	  * @param QString hash
	  * @param QString password
      * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void updateBrutePassword(QString, QString);

 protected:
    void closeEvent(QCloseEvent *event);

 private slots:
    void iconActivated(QSystemTrayIcon::ActivationReason reason);
    void messageClicked(void);
	void showProperties(void);
	
	
	//! Begin Bruteforcer
	/**
	  * Begin Bruteforcer
	  * Initiates the bruteforcing threads and begins communicating with other bruteforcing nodes.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void start(void);
	
	//! Stop Bruteforcer
	/**
	  * Stop Bruteforcer
	  * Stop the bruteforcing threads and close communication with other bruteforcing nodes.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void stop(void);
	
	//! Pause Bruteforcer
	/**
	  * Pause Bruteforcer
	  * Pause the bruteforcing GUI.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void pause(void);

	//! Hash Table Input Hash
	/**
	  * Hash Table Input Hash
	  * Add a single hash from the user to the existing list.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void hashTableInputHash(void);
	
	//! Hash Table Input File
	/**
	  * Hash Table Input File
	  * Add a file of hashes provided by the user to the existing list.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void hashTableInputFile(void);
	
	//! Hash Table Delete Hash
	/**
	  * Hash Table Delete Hash
	  * Remove a hash from the existing list selected by the user.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void hashTableDeleteHash(void);
	
	//! Hash Table Clear
	/**
	  * Hash Table Clear
	  * Remove all hashes from the existing list.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void hashTableClear(void);

	//! Password Save File Table Clear
	/**
	  * Password Save File Table Clear
	  * Clear all hashes that have been added to our saved passwords list.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
	void PasswordSaveFileTableClear(void);
	
	//! Password Save File Table
	/**
	  * Password Save File Table
	  * Save all hashes that have been added to our saved passwords list.
	  * @see bl4ckJack()
      * @see ~bl4ckJack()
      * @return None
	  */
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

//! Thread for reading hashes from a file
/**
 * Class responsible for running under a new thread and reading hashes from a file.
 */
 
class InputHashWorker : public QThread {
	 Q_OBJECT

public:

	//! Class responsible for running under a new thread and reading hashes from a file.
	/**
	 * Class responsible for running under a new thread and reading hashes from a file.
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
     * @return None
	 */
	InputHashWorker(QObject *parent) : QThread(parent) { 
	}

	//! Set File
	/**
	 * Set File
	 * Configure the file that will be processed.
	 * @param QString file name
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
     * @return None
	 */
	void setFile(QString filename) {
		this->filename = filename;
	}

	//! msleep - Sleep for provided milliseconds.
	/**
	 * msleep
	 * Sleep for provided milliseconds.
	 * @param unsigned long time length
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
     * @return None
	 */
	void msleep(unsigned long x) {
		return QThread::msleep(x);
	}

	//! Get Total - Get total amount of time it took hashes to be added by the provided file.
	/**
	 * Get Total
	 * Get total amount of time it took hashes to be added by the provided file.
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
	 * @see this->total;
     * @return qint64
	 */
	qint64 getTotal() {
		return this->total;
	}

	//! Get Current - Get current amount of time it took hashes to be added by the provided file.
	/**
	 * Get Current
	 * Get current amount of time it took hashes to be added by the provided file.
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
     * @return qint64
	 */
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

	//! Extract Hashes From File
	/**
	 * Extract Hashes From File
	 * Extract hashes quickly from the file provided.
	 * @see class InputHashWorker
     * @see ~InputHashWorker()
     * @return None
	 */
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
