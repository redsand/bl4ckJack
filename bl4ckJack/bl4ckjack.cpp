
#include "bl4ckjack.h"
#include "ui_bl4ckjack.h"
#include "bl4ckJack_timer.h"
#include "bl4ckJack_tableview.h"
#include "bl4ckJack_distributed.h"

#include <QDir>

QSettings *settings=NULL;

QList<bl4ckJackModuleList *> bl4ckJackModules;


bl4ckJack::bl4ckJack(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
	isVis = true;
	settings = new QSettings("bl4ck","bl4ckJack");
    createSystemTrayActions();
    createTrayIcon();
	
	tblHashView = new TableModel(ui.tblHash);
	QSortFilterProxyModel *proxyModel = new QSortFilterProxyModel(ui.tblHash);

    proxyModel->setSourceModel(tblHashView);
    proxyModel->setDynamicSortFilter(true);
    ui.tblHash->setSortingEnabled(true);
    ui.tblHash->setSelectionBehavior(QAbstractItemView::SelectRows);

	ui.tblHash->setModel(proxyModel);
    
    ui.tblHash->horizontalHeader()->setStretchLastSection(true);
    ui.tblHash->verticalHeader()->hide();
    ui.tblHash->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui.tblHash->setSelectionMode(QAbstractItemView::SingleSelection);
		

    connect((const QObject *)trayIcon, SIGNAL(messageClicked()), this, SLOT(messageClicked()));
    connect((const QObject *)trayIcon, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
            this, SLOT(iconActivated(QSystemTrayIcon::ActivationReason)));

	createUIMenus();
	createMenuActions();
	createStatusBar();

    trayIcon->show();

	setWindowTitle(tr("bl4ckJack "VERSION" - http://blacksecurity.org/bl4ckJack"));

	// gui this
	moduleProcessDir(tr("."));
	moduleProcessDir(tr("..\\Debug\\"));

	bruteThread = new bl4ckJackBrute(this);

}

bl4ckJack::~bl4ckJack()
{

	// include all ptrs
	
    delete this->minimizeAction;
    delete this->restoreAction;
    delete this->quitAction;
	
	delete this->trayIcon;
	delete this->trayIconMenu;
	
	//! Right Click Menu
	delete this->tblHashAddHash;
	delete this->tblHashAddFile;
	delete this->tblHashDelEntry;
	delete this->tblHashClear;
	
	delete this->tblPasswordClear;
	delete this->tblPasswordSaveFile;

	delete this->tblHashMenu;
	delete this->tblPasswordMenu;

	delete this->tblHashView;
	delete this->bruteThread;
}


/******************************************************/
/* SYSTEM TRAY */
/******************************************************/

 void bl4ckJack::setVisible(bool visible)
 {
	 isVis = visible;
     minimizeAction->setEnabled(visible);
     restoreAction->setEnabled(isMaximized() || !visible);
	 QMainWindow::setVisible(visible);
 }

 void bl4ckJack::closeEvent(QCloseEvent *event)
 {
     if (settings->value("config/move_app_system_tray", true).toBool() && 
		 trayIcon->isVisible()) {
		 if(settings->value("config/notify_moving_system_tray", true).toBool()) {
			 QMessageBox::information(this, tr("bl4ckJack"),
									  tr("The program will keep running in the "
										 "system tray. To terminate the program, "
										 "choose <b>Quit</b> in the context menu "
										 "of the system tray entry."));
		 }
         hide();
         event->ignore();
	 } else {
		 writeSettings();
		 event->accept();
	 }
 }

 void bl4ckJack::iconActivated(QSystemTrayIcon::ActivationReason reason)
 {

     switch (reason) {
     case QSystemTrayIcon::Trigger:
		 if( (settings->value("config/move_app_system_tray", true).toBool()) && 
			 (settings->value("config/move_app_system_tray_single_click", true).toBool())
			 ) {
				  (isVis ? hide() : showNormal());
		}
		break;

     case QSystemTrayIcon::DoubleClick:

		if( (settings->value("config/move_app_system_tray", true).toBool()) && 
			 (!settings->value("config/move_app_system_tray_single_click", true).toBool() )
			 ) {
				  (isVis ? hide() : showNormal());
		}

         break;
     default:
         ;
     }
 }

 void bl4ckJack::messageClicked(void)
 {
     QMessageBox::information(0, tr("bl4ckJack"),
                              tr("Sorry, I already gave what help I could.\n"));
 }

 void bl4ckJack::createSystemTrayActions()
 {
     minimizeAction = new QAction(tr("Mi&nimize"), this);
     connect(minimizeAction, SIGNAL(triggered()), this, SLOT(hide()));

     restoreAction = new QAction(tr("&Show"), this);
     connect(restoreAction, SIGNAL(triggered()), this, SLOT(showNormal()));

     quitAction = new QAction(tr("&Quit"), this);
     connect(quitAction, SIGNAL(triggered()), qApp, SLOT(quit()));

 }

 void bl4ckJack::createTrayIcon()
 {
	 bl4ckJackIcon.addFile(QString::fromUtf8("images/bl4ckJack_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
	 if(bl4ckJackIcon.isNull()) {
		QMessageBox::information(this, tr("bl4ckJack"),
			tr("Unable to open images/bl4ckJack_icon.png for system tray icon."));
	 } else {
		 trayIconMenu = new QMenu(this);
		 trayIconMenu->addAction(minimizeAction);
		 trayIconMenu->addAction(restoreAction);
		 trayIconMenu->addSeparator();
		 trayIconMenu->addAction(quitAction);

		 trayIcon = new QSystemTrayIcon(this);
		 trayIcon->setContextMenu(trayIconMenu);
     
		 trayIcon->setIcon(bl4ckJackIcon);
		 setWindowIcon(bl4ckJackIcon);
		 trayIcon->setToolTip("bl4ckJack "VERSION "\r\nNot Running\r\n");
	 }
 }

 /******************************************************/
 /* Features */
 /******************************************************/

 void bl4ckJack::start(void) {
	ui.actionS_top->setEnabled(true);
	ui.action_Pause->setEnabled(true);
	ui.action_Start->setEnabled(false);
	trayIcon->setToolTip("bl4ckJack "VERSION "\r\nStarting...\r\n");
	repaint();

	connect(bruteThread, SIGNAL(updateBruteStatus(int, int, QString)),
            this, SLOT(updateBruteStatus(int, int, QString)), Qt::QueuedConnection);
	connect(bruteThread, SIGNAL(updateBruteLabels(double, QString, qint64)),
            this, SLOT(updateBruteLabels(double, QString, qint64)), Qt::QueuedConnection);
	connect(bruteThread, SIGNAL(updateBrutePassword(QString, QString)),
            this, SLOT(updateBrutePassword(QString, QString)), Qt::QueuedConnection);

	// start our system thread for brute forcing

	// if we have no hashes loaded, prompt for server listen mode and gather input for socket bind

	if(this->tblHashView->getList().count() == 0) {
		QMessageBox::StandardButton reply = QMessageBox::question(this, tr("bl4ckJack"),
                                     "Would you like to enable the distributed server and wait for clients?",
                                     QMessageBox::Yes | QMessageBox::No);
		if (reply == QMessageBox::Yes) {
			bool ok;
			QString hashEntry = QInputDialog::getText(this, tr("bl4ckJack"), tr("Listen Parameters"), QLineEdit::Normal, tr("0.0.0.0:%1").arg(DEFAULT_PORT), &ok);
			 if(ok && !hashEntry.isEmpty()) {
				 bruteThread->setListenHost(hashEntry);
				 bruteThread->setListenOnly(true);

				trayIcon->setToolTip(tr("bl4ckJack "VERSION "\r\nListening on %1...\r\n").arg(hashEntry));
				repaint();
			 }
		} else {
			bruteThread->setListenHost(tr("127.0.0.1:%1").arg(DEFAULT_PORT));
			bruteThread->setListenOnly(false);
		}

	} else {
		bruteThread->setListenHost(tr("127.0.0.1:%1").arg(DEFAULT_PORT));
		bruteThread->setListenOnly(false);

		// we need to configure what our enabled module is using
		QMap< QString, int> map; 
		QList< QPair<QString, QString> > hashlst = this->tblHashView->getList();
		for(int j=0; j < hashlst.count(); j++)
			map[hashlst.at(j).first]++;

		QStringList lst;
		if(map.count() > 1) {

			QMapIterator<QString, int> i(map);
			while (i.hasNext()) {
				i.next();
				lst.push_back(i.key());
			}

			bool ok = FALSE;
		    QString res = QInputDialog::getItem(this, tr("bl4ckJack"), tr( "Please select a hash module for cracking." ), lst, 0, false, &ok );
			if ( ok ) {
				bruteThread->setModule(res);
			} else {
				bruteThread->setModule(lst.at(0));

				QMessageBox::StandardButton reply = QMessageBox::information(this, tr("bl4ckJack"),
                                     tr("The module '%1' has been selected for module cracking.").arg(lst.at(0)));

			}

		} else if (map.count() == 1) {
			
			QMapIterator<QString, int> i(map);
			if (i.hasNext()) {
				i.next();
				bruteThread->setModule(i.key());
			}
			qDebug() << "Enabling module bruteforce: " << i.key();
		}
	
	}

	bruteThread->start();
 }
 
 void bl4ckJack::stop(void) {
	ui.actionS_top->setEnabled(false);
	ui.action_Pause->setEnabled(false);
	ui.action_Start->setEnabled(true);
	
	trayIcon->setToolTip("bl4ckJack "VERSION "\r\nStopping...\r\n");
	this->statusBar()->showMessage("Stopping...");
	this->statusBar()->repaint();
	repaint();

	bruteThread->terminate();

	while(bruteThread->isRunning() && !bruteThread->isFinished()) {
		bruteThread->msleep(500);
		repaint();
	}

	delete bruteThread;
	bruteThread = new bl4ckJackBrute(this);

	
	QString q;
	q.sprintf("%.6f", 0.0);
	this->ui.lblPPS->setText(tr("%1 Mil/sec").arg(q));
	this->ui.lblTotalHashes->setText(tr("0"));
	this->ui.lblRecoveredPasswords->setText(tr("0"));
	this->ui.lblCompletionTime->setText(tr(""));
	this->ui.progressBar->setValue(0);

	// prompt to save before clear
	if(this->ui.tblPassword->rowCount() > 0) {
		QMessageBox::StandardButton reply = QMessageBox::question(this, tr("bl4ckJack"),
										 "Would you like to save the existing cracked hashes before continuing?",
										 QMessageBox::Yes | QMessageBox::No);
		if (reply == QMessageBox::Yes) {
			PasswordSaveFileTable();
		}
	}
	
	for(int i=ui.tblPassword->rowCount()-1; i >= 0; i--) {
		qDebug() << "removing row: " << i;
		ui.tblPassword->removeRow(i);
	}

	trayIcon->setToolTip("bl4ckJack "VERSION "\r\nStopped.\r\n");
	this->statusBar()->showMessage("Stopped.");
	this->statusBar()->repaint();
	repaint();
 }

 void bl4ckJack::pause(void) {
	ui.actionS_top->setEnabled(true);
	ui.action_Pause->setEnabled(false);
	ui.action_Start->setEnabled(true);
	bruteThread->pause();
	trayIcon->setToolTip("bl4ckJack "VERSION "\r\nPaused...\r\n");
	repaint();
 }

 /******************************************************/
 /* Menus */
 /******************************************************/
 void bl4ckJack::createStatusBar()
 {
     statusBar()->showMessage(tr("bl4ckJack " VERSION " - Ready."));
 }

 void bl4ckJack::hashTableInputHash(void) {
	 bool ok;
	 QString hashEntry = QInputDialog::getText(this, tr("bl4ckJack"), tr("Input Hash String"), QLineEdit::Normal, "", &ok);
	 if(ok && !hashEntry.isEmpty()) {
		
		 //qDebug() << tr("User Input: ") << hashEntry;

		bl4ckJackModuleList *s = NULL;
		
		foreach( s, bl4ckJackModules ) {
		
			if(s->pfbl4ckJackMatch(hashEntry.toAscii()) == true) {
				tblHashView->addEntry(s->moduleInfo->name, hashEntry);
				break;
			} else {
				QMessageBox::critical(this, tr("bl4ckJack"),
					tr("Unable to match input hash against available module: ", hashEntry.toAscii()));
			}
		}
	 } else qDebug() << tr("No input provided.");

 }

 InputHashWorker *worker=NULL;
 
 void bl4ckJack::hashTableInputFile(void) {

	 QString msg;
	 QString filename = QFileDialog::getOpenFileName( 
        this, 
        tr("Open Hash File"), 
        QDir::currentPath(), 
        tr("Hash files (*.chr *.txt *.pwl *.lst);All files (*.*)") );


    if( !filename.isNull() )
    {

		if(worker)
			worker->exit(0);
		delete worker;

		
		trayIcon->setToolTip("bl4ckJack "VERSION "\r\nLoading input hashes from file...\r\n");
		msg = "Loading input hashes from file: ";
		msg.append(filename);
		this->statusBar()->showMessage(msg, 20000);

		worker = new InputHashWorker(this);
		worker->setFile(filename);

		connect(worker, SIGNAL(updateUIFileAdd(QString, QString, QString, float)),
            this, SLOT(updateUIFileAdd(QString, QString, QString, float)), Qt::QueuedConnection);

		ui.tblHash->setUpdatesEnabled(false);
		worker->start();
		float tmpval = 0;
		while(!worker->isFinished()) {
			tmpval = worker->getCurrent();
			tmpval = tmpval / worker->getTotal();
			tmpval = tmpval * 100.0;
			ui.progressBar->setValue(tmpval);
			worker->msleep(500);
		}
		worker->wait();

		ui.tblHash->setUpdatesEnabled(true);
		ui.tblHash->resizeColumnsToContents();
		ui.tblHash->horizontalHeader()->setStretchLastSection(true);
		ui.tblHash->repaint();

		worker->exit(0);
		delete worker;
		worker = NULL;

		trayIcon->setToolTip("bl4ckJack "VERSION "\r\nLoading input hashes from file completed.\r\n");
		ui.progressBar->setValue(100);
		msg = "Successfully loaded input hashes from file: ";
		msg.append(filename);
		this->statusBar()->showMessage(msg);

    } else qDebug() << tr("No input provided.");
	

 }

 void bl4ckJack::hashTableDeleteHash(void) {
	 tblHashView->removeRow( ui.tblHash->currentIndex().row());
 }

 void bl4ckJack::hashTableClear(void) {
	// qDebug() << "removing row count " << ui.tblHash->rowCount();
	int i=0;
	tblHashView->removeRows(0, tblHashView->rowCount(QModelIndex()));
 }
 
 void bl4ckJack::PasswordSaveFileTableClear(void) {
	 
	qDebug() << "removing row count " << ui.tblPassword->rowCount();
	int i=0;
	for(i=ui.tblPassword->rowCount()-1; i >= 0; i--) {
		qDebug() << "removing row: " << i;
		ui.tblPassword->removeRow(i);
	}
 }

 void bl4ckJack::PasswordSaveFileTable(void) {
	 /*
	qDebug() << "saving row count " << ui.tblPassword->rowCount();
	*/
	int i=0;
	QString filename = QFileDialog::getSaveFileName( 
        this, 
        tr("Save Hash File"), 
        QDir::currentPath(), 
        tr("CSV Database (*.csv);;All files (*.*)") );

    if( !filename.isNull() )
    {
		QFile file(filename);
		// open file contents, put contents into textbox
	    if (file.open(QFile::WriteOnly)) {
			QTextStream out(&file);

			for(i=0; i < ui.tblPassword->rowCount(); i++) {
				qDebug() << "saving row: " << i;
				out << ui.tblPassword->item(i,0)->text().toLatin1().data() << ", " << ui.tblPassword->item(i,1)->text().toLatin1().data() << ", " << ui.tblPassword->item(i,2)->text().toLatin1().data() << endl;
				//ui.tblPassword->removeRow(i);
			}

			file.close();
		}
	}
 }

 void bl4ckJack::createUIMenus() {
	
	QAction *sep = NULL;

	tblHashAddHash = new QAction(tr("&Add Hash"), this);
    connect(tblHashAddHash, SIGNAL(triggered()), this, SLOT(hashTableInputHash()));

	tblHashAddFile = new QAction(tr("Add &File"), this);
    connect(tblHashAddFile, SIGNAL(triggered()), this, SLOT(hashTableInputFile()));

	tblHashDelEntry = new QAction(tr("&Remove Entry"), this);
    connect(tblHashDelEntry, SIGNAL(triggered()), this, SLOT(hashTableDeleteHash()));

	tblHashClear = new QAction(tr("&Clear"), this);
    connect(tblHashClear, SIGNAL(triggered()), this, SLOT(hashTableClear()));
	
	ui.tblHash->setContextMenuPolicy(Qt::ActionsContextMenu);
	ui.tblHash->addAction(tblHashAddHash);
	ui.tblHash->addAction(tblHashAddFile);


	sep = new QAction(this);
	sep->setSeparator(true);
	ui.tblHash->addAction(sep);

	ui.tblHash->addAction(tblHashDelEntry);

	sep = new QAction(this);
	sep->setSeparator(true);
	ui.tblHash->addAction(sep);
	
	ui.tblHash->addAction(tblHashClear);

	/***************************************/

	tblPasswordClear = new QAction(tr("&Clear"), this);
    connect(tblPasswordClear, SIGNAL(triggered()), this, SLOT(PasswordSaveFileTableClear()));

	tblPasswordSaveFile = new QAction(tr("&Save to File"), this);
    connect(tblPasswordSaveFile, SIGNAL(triggered()), this, SLOT(PasswordSaveFileTable()));
	
	ui.tblPassword->setContextMenuPolicy(Qt::ActionsContextMenu);

	ui.tblPassword->addAction(tblPasswordSaveFile);
	
	sep = new QAction(this);
	sep->setSeparator(true);
	ui.tblPassword->addAction(sep);

	ui.tblPassword->addAction(tblPasswordClear);

 }

 void bl4ckJack::showProperties(void) {
	 
	ConfigDialog cfgui(this);
	cfgui.show();
	cfgui.setFocus();
	cfgui.exec();
 }

 void bl4ckJack::createMenuActions() {

	
	connect(ui.action_Properties, SIGNAL(triggered()), this, SLOT(showProperties()));
	connect(ui.action_About, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.action_Close, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.action_New, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.action_Open, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.action_Pause, SIGNAL(triggered()), this, SLOT(pause()));
	connect(ui.action_Save, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.action_Start, SIGNAL(triggered()), this, SLOT(start()));
	connect(ui.action_Updates, SIGNAL(triggered()), this, SLOT(hide()));
	connect(ui.actionS_top, SIGNAL(triggered()), this, SLOT(stop()));
	connect(ui.actionSave_As, SIGNAL(triggered()), this, SLOT(hide()));

	connect(ui.action_Quit, SIGNAL(triggered()), qApp, SLOT(quit()));

    ui.actionS_top->setEnabled(false);
	ui.action_Pause->setEnabled(false);
	ui.action_Start->setEnabled(true);
 }


 void bl4ckJack::updateBruteStatus(int priority, int status_bar, QString b) {

	QString tmp;

	switch(priority) {

		case 1:
			// fatal error popup msg
			tmp.append("Fatal error: ");
			tmp.append(b);
			
			QMessageBox::critical(0, tr("bl4ckJack"), tmp);
			this->statusBar()->showMessage(tmp);
			this->statusBar()->repaint();
			this->stop();
			break;
		case 2: // warning popup msg
			
			tmp.append("Warning: ");
			tmp.append(b);
			QMessageBox::warning(0, tr("bl4ckJack"), tmp);
			this->statusBar()->showMessage(tmp);
			this->statusBar()->repaint();
			break;
		case 3:
			tmp.append(b);
			//QMessageBox::warning(0, tr("bl4ckJack"), tmp);
			this->statusBar()->showMessage(tmp);
			this->statusBar()->repaint();
			break;
	}

	ui.progressBar->setValue(status_bar);
	ui.progressBar->repaint();


 }

//static int asdlafjsd=0;

void bl4ckJack::updateUIFileAdd(QString a, QString b, QString c, float status) {
	 
	QString tmp;

	tmp = "Added ";
	tmp.append(a);
	tmp.append(" hash ");
	tmp.append(c);
	tmp.append(" successfully.");

	this->statusBar()->showMessage(tmp);
	this->statusBar()->repaint();
	this->tblHashView->addEntry(a, c);
	this->ui.lblTotalHashes->setText(tr("%1").arg(this->tblHashView->getList().count()));
	//this->repaint();

	/*
	if(asdlafjsd % 10 == 0) {
		QCoreApplication::processEvents();
		asdlafjsd = 0;
	}
	*/

 }

void bl4ckJack::updateBruteLabels(double milpw, QString ttl, qint64 crackedPasswords) {
	
	QString q, q2, q3;
	if(milpw > 0) {
		q.sprintf("%.6f", milpw);
		this->ui.lblPPS->setText(tr("%1 Mil/sec").arg(q));
		trayIcon->setToolTip(tr("bl4ckJack "VERSION "\r\nBruteforcing @ %1 Mil/sec\r\n").arg(q));
	}

	if(crackedPasswords > 0) {
		q2.sprintf("%ld",this->tblHashView->getList().count() - crackedPasswords);
		this->ui.lblTotalHashes->setText(q2);
		q3.sprintf("%ld",crackedPasswords);
		this->ui.lblRecoveredPasswords->setText(q3);

		if( crackedPasswords == this->tblHashView->getList().count()) {
			// we're finished!
			this->stop();
		}
	}
	this->ui.lblCompletionTime->setText(ttl);
}

void bl4ckJack::updateBrutePassword(QString hash, QString password) {

	QTableWidgetItem *item=new QTableWidgetItem (this->bruteThread->getModule());
	QTableWidgetItem *item2=new QTableWidgetItem (password);
	QTableWidgetItem *item3=new QTableWidgetItem (hash);

	this->ui.tblPassword->insertRow(this->ui.tblPassword->rowCount());
	this->ui.tblPassword->setItem(this->ui.tblPassword->rowCount() - 1, 0, item);
	this->ui.tblPassword->setItem(this->ui.tblPassword->rowCount() - 1, 1, item2);
	this->ui.tblPassword->setItem(this->ui.tblPassword->rowCount() - 1, 2, item3);

	this->ui.tblPassword->resizeColumnsToContents();

}
 
 /******************************************************/
 /* Configuration */
 /******************************************************/

 void bl4ckJack::writeSettings() {
	// settings->beginGroup("config");
 }

 /******************************************************/
 /* Modules */
 /******************************************************/
 void bl4ckJack::moduleProcessDir(QString entry) {
	
	 QDir myDir(entry);

#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	QStringList filter;
	filter << "*.dll" << "*.so";
#else
	QStringList filter;
	filter << "*.so" << "*.a" << "*.sl" << "*.dylib" << "*.bundle" ;
#endif

	QStringList list = myDir.entryList();

	int i=0;

	while (!bl4ckJackModules.isEmpty())
		free((void *)bl4ckJackModules.takeFirst());

	bl4ckJackModules.clear();

	qDebug() << "Processing directory for modules: " << entry;

	for(i=0; i < list.size(); i++) {
		// try and load
		// if load and module exists and our module funcs exist:
		//		map into struct module list
		qDebug() << "Found file within directory: " << list[i];

		QLibrary library(list[i]);
		if(library.load()) {
			// if our funcs don't exist
			// library.unload();

			fbl4ckJackInit pfbl4ckJackInit = (fbl4ckJackInit) library.resolve("bl4ckJackInit");

			fbl4ckJackMatch pfbl4ckJackMatch = (fbl4ckJackMatch) library.resolve("bl4ckJackMatch");
			
			fbl4ckJackInfo pfbl4ckJackInfo = (fbl4ckJackInfo) library.resolve("bl4ckJackInfo");
			
			fbl4ckJackFree pfbl4ckJackFree = (fbl4ckJackFree) library.resolve("bl4ckJackFree");
			
			fbl4ckJackGenerate pfbl4ckJackGenerate = (fbl4ckJackGenerate) library.resolve("bl4ckJackGenerate");

			if(pfbl4ckJackInit)
			qDebug() << " pfbl4ckJackInit Found!";
			else
			qDebug() << " pfbl4ckJackInit Not Found";

			if(pfbl4ckJackMatch)
			qDebug() << " pfbl4ckJackMatch Found!";
			else
			qDebug() << " pfbl4ckJackMatch Not Found";

			if(pfbl4ckJackInfo)
			qDebug() << " pfbl4ckJackInfo Found!";
			else
			qDebug() << " pfbl4ckJackInfo Not Found";

			if(pfbl4ckJackFree)
			qDebug() << " pfbl4ckJackFree Found!";
			else
			qDebug() << " pfbl4ckJackFree Not Found";

			if(pfbl4ckJackGenerate)
			qDebug() << " pfbl4ckJackGenerate Found!";
			else
			qDebug() << " pfbl4ckJackGenerate Not Found";

			if(pfbl4ckJackInit && pfbl4ckJackMatch && pfbl4ckJackInfo && pfbl4ckJackFree
				&& pfbl4ckJackGenerate) {
				bl4ckJackModuleList *bjm = (bl4ckJackModuleList *) malloc(sizeof(bl4ckJackModuleList)); // new bl4ckJackModuleList;
				bjm->pfbl4ckJackInit = pfbl4ckJackInit;
				bjm->pfbl4ckJackMatch = pfbl4ckJackMatch;
				bjm->pfbl4ckJackInfo = pfbl4ckJackInfo;
				bjm->pfbl4ckJackFree = pfbl4ckJackFree;
				bjm->pfbl4ckJackGenerate = pfbl4ckJackGenerate;
				bjm->moduleInfo = pfbl4ckJackInfo();

				bl4ckJackModules.append(bjm);
				qDebug() << "Detected bl4ckJack module: " << bjm->moduleInfo->name << " (" << bjm->moduleInfo->authors << ")";

			} else {
				qDebug() << "Failed to identify functions.";
				library.unload();
			}
		}
	}
 }
