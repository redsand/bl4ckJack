#include <QtGui>
#include <QIODevice>
#include "bl4ckJack_config.h"
#include "cuda_gpu.h"

ConfigurationPage *confPage = NULL;
GPUPage	*gpuPage = NULL;
DCPage *dcPage = NULL;
ModulesPage	*modPage = NULL;

 ConfigurationPage::ConfigurationPage(QWidget *parent)
     : QWidget(parent)
 {

     QGroupBox *bruteGroup = new QGroupBox(tr("Bruteforce Options"));
     QLabel *charsetLabel = new QLabel(tr("Charset:"));
	 
	 charsetText = new QPlainTextEdit(settings->value("config/charset",tr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")).toString()); // config/charset
	 charsetText->setFocus();
	 charsetText->setFixedHeight(40);

     QHBoxLayout *charsetLayout = new QHBoxLayout;
     charsetLayout->addWidget(charsetLabel);
     charsetLayout->addWidget(charsetText);
	 

	 QPushButton *p = new QPushButton(tr("Open Charset"),this); 



     QVBoxLayout *bruteLayout = new QVBoxLayout;
     bruteLayout->addLayout(charsetLayout);
	 bruteLayout->addWidget(p);
     bruteGroup->setLayout(bruteLayout);
	 connect(p, SIGNAL(clicked()),this, SLOT(pushOpenCharset()));

	
	 /*****************************************************************/

	 QGroupBox *cfgGroup = new QGroupBox(tr("Configuration"));
	 hideWindowCheckBox = new QCheckBox(tr("Notify me when moving to system tray")); // config/notify_moving_system_tray
	 closeWindowCheckBox = new QCheckBox(tr("Move existing application to system tray")); // config/move_app_system_tray
	 close2WindowCheckBox = new QCheckBox(tr("Single-click toggle moving application to system tray")); // config/move_app_system_tray_single_click
	 notifySaveFileOverwriteCheckBox = new QCheckBox(tr("Prompt when overwriting saved session file")); // config/session_overwrite_autosave
	 autoWindowCheckBox = new QCheckBox(tr("Enable automated saving of session")); // config/session_disable_autosave

 	 connect(closeWindowCheckBox, SIGNAL(toggled(bool)), this, SLOT(commitSysTrayData(bool)));

 	 connect(autoWindowCheckBox, SIGNAL(toggled(bool)), this, SLOT(commitBackupData(bool)));


	 if(settings->value("config/move_app_system_tray",true).toBool()) {
		closeWindowCheckBox->toggle();

		 hideWindowCheckBox->setEnabled(true);
		 close2WindowCheckBox->setEnabled(true);

		 if(settings->value("config/notify_moving_system_tray",true).toBool()) {
			hideWindowCheckBox->toggle();
		 }

		 if(settings->value("config/move_app_system_tray_single_click",true).toBool()) {
			close2WindowCheckBox->toggle();
		 }
	 } else {
		 hideWindowCheckBox->setEnabled(false);
		 close2WindowCheckBox->setEnabled(false);
	 }

	 if(settings->value("config/session_disable_autosave",false).toBool()) {
		autoWindowCheckBox->toggle();
	 
		 if(settings->value("config/session_overwrite_autosave",true).toBool()) {
			notifySaveFileOverwriteCheckBox->toggle();
		 }
	 } else {
		 notifySaveFileOverwriteCheckBox->setEnabled(false);
	 }

     QVBoxLayout *cfgLayout = new QVBoxLayout;

	 cfgLayout->addWidget(closeWindowCheckBox);
	 cfgLayout->addWidget(hideWindowCheckBox);
	 cfgLayout->addWidget(close2WindowCheckBox);
	 cfgLayout->addWidget(autoWindowCheckBox);
	 cfgLayout->addWidget(notifySaveFileOverwriteCheckBox);
	
	 cfgGroup->setLayout(cfgLayout);
	 
	 /*****************************************************************/

     QVBoxLayout *mainLayout = new QVBoxLayout;
     mainLayout->addWidget(bruteGroup);
	 mainLayout->addWidget(cfgGroup);
     mainLayout->addStretch(1);
     setLayout(mainLayout);
 }

 ConfigurationPage::~ConfigurationPage(void) {
	delete hideWindowCheckBox;
	delete closeWindowCheckBox;
	delete close2WindowCheckBox;
	delete notifySaveFileOverwriteCheckBox;
	delete autoWindowCheckBox;
	delete charsetText;
 }

 void ConfigurationPage::pushOpenCharset() {
	 QString filename = QFileDialog::getOpenFileName( 
        this, 
        tr("Open Charset File"), 
        QDir::currentPath(), 
        tr("Charset files (*.chr *.txt);;All files (*.*)") );

    if( !filename.isNull() )
    {
		QFile file(filename);
		// open file contents, put contents into textbox
	    if (file.open(QFile::ReadOnly)) {
			 char buf[1024];
			 qint64 lineLength = 0;
			 charsetText->setPlainText(tr(""));
			 while(file.readLine(buf, sizeof(buf)) >= 0) {
				charsetText->appendPlainText(buf);
			 }
			file.close();
		}
    }

 }

 void ConfigurationPage::commitSysTrayData(bool value) {
	// if not checked, disable:
		// hideWindowCheckBox
		// close2WindowCheckBox
	 if(value) {
	 //if(closeWindowCheckBox->isChecked()) {
		hideWindowCheckBox->setEnabled(true);
		close2WindowCheckBox->setEnabled(true);
	 } else {	
		hideWindowCheckBox->setEnabled(false);
		close2WindowCheckBox->setEnabled(false);
	 }
 }

 void ConfigurationPage::commitBackupData(bool value) {
	// if not checked, disable:
		// notifySaveFileOverwriteCheckBox
	 if(value) {
		notifySaveFileOverwriteCheckBox->setEnabled(true);
	 } else {	
		notifySaveFileOverwriteCheckBox->setEnabled(false);
	 }
 }

 GPUPage::GPUPage(QWidget *parent)
     : QWidget(parent)
 {

	 GPU_Dev g;
	 int i;
	 char buf[1024];

     QGroupBox *gpuGroup = new QGroupBox(tr("Available System GPUs"));
     gpuList = new QListWidget();
	 gpuList->setSortingEnabled(1);
	 gpuList->setFixedHeight(160);

	 for(i=0; i < g.getDevCount(); i++) {
		g.getDevInfoStr(i, buf, sizeof(buf)-1);
		QListWidgetItem *itm = new QListWidgetItem(buf, gpuList);
		itm->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
		itm->setCheckState( (settings->value(tr("config/gpu_device_%1_enabled").arg(i),true).toBool()) ? Qt::Checked : Qt::Unchecked);
		gpuList->insertItem(i + 0, itm);
	 }
	 if(i == 0) {
		QListWidgetItem *itm = new QListWidgetItem(tr("(No CUDA/GPU device available.)"), gpuList);
		gpuList->insertItem(i + 0, itm);
		gpuList->setEnabled(false);
	 }

	 
     QVBoxLayout *gpuLayout = new QVBoxLayout;
	 gpuLayout->addWidget(gpuList);
	 gpuGroup->setLayout(gpuLayout);
	
	 /*****************************************************************/

	 QGroupBox *cfgGroup = new QGroupBox(tr("GPU Configuration"));
	 
// GPUPage for following options:
	inputMaximumLoops = new QLineEdit(settings->value("config/gpu_maximum_loops",1024).toString());
	inputMaximumLoops->setMaximumWidth(64);
	inputMaximumLoops->setMinimumWidth(64);

	inputMaxMemInit = new QLineEdit(settings->value("config/gpu_max_mem_init",8000640).toString());
	inputMaxMemInit->setMaximumWidth(100);
	inputMaxMemInit->setMinimumWidth(100);

	inputGPUThreads = new QLineEdit(settings->value("config/gpu_thread_count",512).toString());
	inputGPUThreads->setMaximumWidth(64);
	inputGPUThreads->setMinimumWidth(64);
	
	inputEnableHardwareMonitoring = new QCheckBox(tr("Enable GPU Hardware Health Monitoring"));
	if(settings->value("config/gpu_health_monitor_enabled").toBool()) {
		inputEnableHardwareMonitoring->setChecked(true);
	} else
		inputEnableHardwareMonitoring->setChecked(false);

     QVBoxLayout *cfgLayout = new QVBoxLayout;
	 cfgLayout->setAlignment(Qt::AlignLeft);

	 QLabel *label;
	 QHBoxLayout *cfgLayoutH;
	 
	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(inputMaximumLoops);
	 label = new QLabel(tr("Maximum GPU kernel iterations before refresh"));
	 label->setFixedWidth(350);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);

	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(inputGPUThreads);
	 label = new QLabel(tr("Maximum threads available for use"));
	 label->setFixedWidth(256);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);

	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(inputMaxMemInit);
	 label = new QLabel(tr("Maximum memory available for allocation (bytes)"));
	 label->setFixedWidth(350);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);

	 cfgLayout->addWidget(inputEnableHardwareMonitoring);

	 cfgGroup->setLayout(cfgLayout);
	 
	 /*****************************************************************/

     QVBoxLayout *mainLayout = new QVBoxLayout;
     mainLayout->addWidget(gpuGroup);
	 mainLayout->addWidget(cfgGroup);
     mainLayout->addStretch(1);
     setLayout(mainLayout);
 }

 /*
	- Add/Remove Servers from list
	- Enable Local Server
	- Minimum Keyspace Tokens per Client (default: 10)
	- Enable compression
	- Require SSL
 */

 DCPage::DCPage(QWidget *parent)
     : QWidget(parent)
 {

	 int i=0;
     QGroupBox *serverGroup = new QGroupBox(tr("Distributed Devices"));
     serverList = new QListWidget();
	 serverList->setSortingEnabled(1);
	 serverList->setFixedHeight(80);

	 QStringList lst = settings->value("config/dc_hosts").toStringList();
	 for(i=0; i < lst.count(); i++) {
		qDebug() << " initial adding of " << lst.at(i);
		QListWidgetItem *itm = new QListWidgetItem(lst.at(i), serverList);
		itm->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
		serverList->insertItem(i + 1, itm);
	 }

     QVBoxLayout *serverLayout = new QVBoxLayout;
	 serverLayout->addWidget(serverList);
	 serverGroup->setLayout(serverLayout);
	
	 /*****************************************************************/

	 QGroupBox *cfgGroup = new QGroupBox(tr("Distributed Computing Configuration"));
	 

// GPUPage for following options:
	txtMinimumTokens = new QLineEdit(settings->value("config/dc_minimum_tokens",10).toString());
	txtMinimumTokens->setMaximumWidth(64);
	txtMinimumTokens->setMinimumWidth(64);

	txtCPUTokenPercentage = new QLineEdit(settings->value("config/dc_cpu_keyspace_pct",5).toString());
	txtCPUTokenPercentage->setMaximumWidth(64);
	txtCPUTokenPercentage->setMinimumWidth(64);



	enableLocalServerCheckBox = new QCheckBox(tr("Enable Local Computing Service"));
	if(settings->value("config/dc_local_service").toBool()) {
		enableLocalServerCheckBox->setChecked(true);
	} else
		enableLocalServerCheckBox->setChecked(false);

	
	txtTimeout = new QLineEdit(settings->value("config/dc_timeout",2*1000).toString());
	txtTimeout->setMaximumWidth(64);
	txtTimeout->setMinimumWidth(64);

	enableSSLCheckBox = new QCheckBox(tr("Enable SSL Encryption"));
	if(settings->value("config/dc_ssl_encryption").toBool()) {
		enableSSLCheckBox->setChecked(true);
	} else
		enableSSLCheckBox->setChecked(false);

	enableCompressionCheckBox = new QCheckBox(tr("Enable Stream Compression"));
	if(settings->value("config/dc_compression").toBool()) {
		enableCompressionCheckBox->setChecked(true);
	} else
		enableCompressionCheckBox->setChecked(false);


     QVBoxLayout *cfgLayout = new QVBoxLayout;
	 cfgLayout->setAlignment(Qt::AlignLeft);

	 QLabel *label;
	 QHBoxLayout *cfgLayoutH;
	 
	 cfgLayout->addWidget(enableLocalServerCheckBox);
	 cfgLayout->addWidget(enableCompressionCheckBox);
	 cfgLayout->addWidget(enableSSLCheckBox);

	inputMaxPasswordSize = new QLineEdit(settings->value("config/dc_max_password_size",16).toString());
	inputMaxPasswordSize->setMaximumWidth(64);
	inputMaxPasswordSize->setMinimumWidth(64);

	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(inputMaxPasswordSize);
	 label = new QLabel(tr("Maximum password length"));
	 label->setFixedWidth(256);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);


	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(txtMinimumTokens);
	 label = new QLabel(tr("Minimum Keyset Tokens Per Host"));
	 label->setFixedWidth(256);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);
	 
	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(txtCPUTokenPercentage);
	 label = new QLabel(tr("Percentage of keyspace assigned per CPU"));
	 label->setFixedWidth(256);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);
	 

	 cfgLayoutH = new QHBoxLayout;
	 cfgLayoutH->setAlignment(Qt::AlignLeft);
	 cfgLayoutH->addWidget(txtTimeout);
	 label = new QLabel(tr("Remote Host Connection Timeout"));
	 label->setFixedWidth(256);
	 cfgLayoutH->addWidget(label);
	 cfgLayout->addLayout(cfgLayoutH);

	 cfgGroup->setLayout(cfgLayout);
	 
	 /*****************************************************************/

     QVBoxLayout *mainLayout = new QVBoxLayout;
     mainLayout->addWidget(serverGroup);
	 mainLayout->addWidget(cfgGroup);
     mainLayout->addStretch(1);
     setLayout(mainLayout);

	serverList->setContextMenuPolicy(Qt::CustomContextMenu);
	serverListMenu = new QMenu(this);

	actionAddServer = new QAction(tr("&Add"), this);
	connect(actionAddServer, SIGNAL(triggered()), this, SLOT(actionAddServerFn()));
	serverListMenu->addAction(actionAddServer);
	
	QAction *sep = new QAction(this);
	sep->setSeparator(true);
	serverListMenu->addAction(sep);

	actionDelServer = new QAction(tr("&Remove"), this);
    connect(actionDelServer, SIGNAL(triggered()), this, SLOT(actionDelServerFn()));
	serverListMenu->addAction(actionDelServer);

	connect(serverList, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(customContextMenu(const QPoint &)));
    serverList->setToolTip("Right click for more options.");
 }

 void DCPage::customContextMenu(const QPoint &pos) {
	 QPoint gpos = this->mapToGlobal(pos);
	 this->serverListMenu->exec(gpos);
 }

 void DCPage::actionDelServerFn(void) {
	qDeleteAll(serverList->selectedItems());
	//qDebug() << "del menu";
 }

 void DCPage::actionAddServerFn(void) {
	bool ok;
	int i=0, orig_count = 0;
	QString entry = QInputDialog::getText(this, tr("bl4ckJack"), tr("Remote Host"), QLineEdit::Normal, "", &ok);
	if(ok && !entry.isEmpty()) {
		QStringList lst = settings->value("config/dc_hosts").toStringList();
		orig_count = lst.count();
		qDebug() << "checking hosts list for " << entry;
		if(!lst.contains(entry)) {
			lst.append(entry);
			QListWidgetItem *itm = new QListWidgetItem(entry, serverList);
			itm->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
			serverList->insertItem(lst.count() + 1, itm);
		}
	}
 }

 DCPage::~DCPage() {
	 delete this->serverList;
	 delete this->actionDelServer;
	 delete this->actionAddServer;
	 delete this->enableLocalServerCheckBox;
	 delete this->txtMinimumTokens;
	 delete this->enableSSLCheckBox;
	 delete this->enableCompressionCheckBox;
	 delete inputMaxPasswordSize;
	 delete this->serverListMenu;
 }

 //! Function to automatically read available modules and associated configuration options
 ModulesPage::ModulesPage(QWidget *parent)
     : QWidget(parent)
 {

     QGroupBox *packageGroup = new QGroupBox(tr("Available Modules"));

     QTableWidget *moduleList = new QTableWidget;
	 //moduleList->setFixedHeight(160);
	 QStringList columns;
	 columns << "Module" << "Version" << "Release Date" << "Author" ;
	 moduleList->setColumnCount(4);
	 moduleList->setHorizontalHeaderLabels(columns);

	 bl4ckJackModuleList *s = NULL;
	 int row = moduleList->rowCount();

	 foreach( s, bl4ckJackModules ) {
		qDebug() << "Processing module " << s->moduleInfo->name;
		moduleList->insertRow(row);
		QTableWidgetItem *entry = new QTableWidgetItem;
		entry->setText(s->moduleInfo->name);
		entry->setTextAlignment(Qt::AlignVCenter);
		entry->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
		//entry->setIcon(QIcon(QPixmap(":/Images/cubed.png")));
		moduleList->setItem(row, 0, entry);

		QString temp;
		temp.sprintf("%.2f", s->moduleInfo->version);
		entry = new QTableWidgetItem;
		entry->setText(temp);
		entry->setTextAlignment(Qt::AlignVCenter);
		entry->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled);
		//entry->setIcon(QIcon(QPixmap(":/Images/cubed.png")));
		moduleList->setItem(row, 1, entry);

		
		entry = new QTableWidgetItem;
		entry->setText(s->moduleInfo->date);
		entry->setTextAlignment(Qt::AlignVCenter);
		entry->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled);
		//entry->setIcon(QIcon(QPixmap(":/Images/cubed.png")));
		moduleList->setItem(row, 2, entry);
		
		entry = new QTableWidgetItem;
		entry->setText(s->moduleInfo->authors);
		entry->setTextAlignment(Qt::AlignVCenter);
		entry->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled);
		//entry->setIcon(QIcon(QPixmap(":/Images/cubed.png")));
		moduleList->setItem(row, 3, entry);

		row++;

	 }

	 if(row == 0) {
		QTableWidgetItem *itm = new QTableWidgetItem;
		itm->setText(tr("(No bl4ckJack GPU modules available.)"));
		moduleList->insertRow(row);
		moduleList->setItem(row, 0, itm);
		moduleList->setEnabled(false);
	 }

	 moduleList->resizeColumnsToContents();
	 moduleList->setSortingEnabled(true);
	 moduleList->sortItems(0, Qt::AscendingOrder );
	 // moduleList-> disable row count on left hand side?
	 moduleList->verticalHeader()->setVisible(false);


	 QVBoxLayout *packageLayout = new QVBoxLayout;
     packageLayout->addWidget(moduleList);
     packageGroup->setLayout(packageLayout);

     QVBoxLayout *mainLayout = new QVBoxLayout;
	 mainLayout->addWidget(packageGroup);
     mainLayout->addStretch(1);
     setLayout(mainLayout);
 }

 ConfigDialog::ConfigDialog(QWidget *parent)
 {
     contentsWidget = new QListWidget;
     contentsWidget->setViewMode(QListView::IconMode);
     contentsWidget->setIconSize(QSize(96, 84));
     contentsWidget->setMovement(QListView::Static);
     contentsWidget->setMaximumWidth(128);
     contentsWidget->setSpacing(12);

     pagesWidget = new QStackedWidget;
	 
	 //if(!confPage) 
	 {
		confPage = new ConfigurationPage;
	 }

	 //if(!modPage) 
	 {
		modPage = new ModulesPage;
	 }

	 {
		gpuPage = new GPUPage;
	 }

	 {
		dcPage = new DCPage;
	 }

     pagesWidget->addWidget(confPage);
     pagesWidget->addWidget(gpuPage);
     pagesWidget->addWidget(modPage);
     pagesWidget->addWidget(dcPage);

     QPushButton *closeButton = new QPushButton(tr("Close"));

     createIcons();
     contentsWidget->setCurrentRow(0);

     connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));

     QHBoxLayout *horizontalLayout = new QHBoxLayout;
     horizontalLayout->addWidget(contentsWidget);
     horizontalLayout->addWidget(pagesWidget, 1);

     QHBoxLayout *buttonsLayout = new QHBoxLayout;
     buttonsLayout->addStretch(1);
     buttonsLayout->addWidget(closeButton);

     QVBoxLayout *mainLayout = new QVBoxLayout;
     mainLayout->addLayout(horizontalLayout);
     mainLayout->addStretch(1);
     mainLayout->addSpacing(12);
     mainLayout->addLayout(buttonsLayout);
     setLayout(mainLayout);

     setWindowTitle(tr("Properties"));
	 QIcon icon;
	 icon.addFile(QString::fromUtf8("images/bl4ckJack_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
	 setWindowIcon(icon);
 }

 void ConfigDialog::createIcons()
 {
	 QListWidgetItem *configButton = new QListWidgetItem(contentsWidget);
     configButton->setIcon(QIcon("images/configuration_icon.png"));
     configButton->setText(tr("Setup"));
     configButton->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
     configButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

	 QListWidgetItem *gpuButton = new QListWidgetItem(contentsWidget);
     gpuButton->setIcon(QIcon("images/gpu_icon.png"));
     gpuButton->setText(tr("GPU"));
	 gpuButton->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
     gpuButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

	 QListWidgetItem *moduleButton = new QListWidgetItem(contentsWidget);
     moduleButton->setIcon(QIcon("images/module_icon.png"));
     moduleButton->setText(tr("Modules"));
	 moduleButton->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
     moduleButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

	 QListWidgetItem *dcButton = new QListWidgetItem(contentsWidget);
     dcButton->setIcon(QIcon("images/distributed_computing_icon.png"));
     dcButton->setText(tr("Distribution"));
	 dcButton->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
     dcButton->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

     connect(contentsWidget,
             SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
             this, SLOT(changePage(QListWidgetItem*,QListWidgetItem*)));
 }

 void ConfigDialog::changePage(QListWidgetItem *current, QListWidgetItem *previous)
 {
     if (!current)
         current = previous;

     pagesWidget->setCurrentIndex(contentsWidget->row(current));
 }

 
//! Overloaded functions: read our options and set our settings accordingly
 void ConfigDialog::closeEvent(QCloseEvent *event) {

	 settings->setValue("config/notify_moving_system_tray", confPage->hideWindowCheckBox->isChecked());
	 settings->setValue("config/move_app_system_tray", confPage->closeWindowCheckBox->isChecked());
	 settings->setValue("config/move_app_system_tray_single_click", confPage->close2WindowCheckBox->isChecked());
	 settings->setValue("config/session_overwrite_autosave", confPage->notifySaveFileOverwriteCheckBox->isChecked());
	 settings->setValue("config/session_disable_autosave", confPage->autoWindowCheckBox->isChecked());

	 settings->setValue("config/charset", confPage->charsetText->toPlainText());

	 for(int i=0; i < gpuPage->gpuList->count(); i++) {
		 QListWidgetItem *itm = gpuPage->gpuList->item(i);
		 settings->setValue(tr("config/gpu_device_%1_enabled").arg(i), (itm->checkState() == Qt::Checked) ? true : false);
	 }

	 settings->setValue("config/gpu_maximum_loops", gpuPage->inputMaximumLoops->text().toInt());
	 settings->setValue("config/gpu_max_mem_init", gpuPage->inputMaxMemInit->text().toLong());
	 settings->setValue("config/gpu_health_monitor_enabled", gpuPage->inputEnableHardwareMonitoring->isChecked());
	 settings->setValue("config/gpu_thread_count", gpuPage->inputGPUThreads->text().toLong());


	 settings->setValue("config/dc_max_password_size", dcPage->inputMaxPasswordSize->text().toInt());
	 settings->setValue("config/dc_local_service", dcPage->enableLocalServerCheckBox->isChecked());
	 settings->setValue("config/dc_compression", dcPage->enableCompressionCheckBox->isChecked());
	 settings->setValue("config/dc_ssl_encryption", dcPage->enableSSLCheckBox->isChecked());
	 settings->setValue("config/dc_minimum_tokens", dcPage->txtMinimumTokens->text().toLong());
	 settings->setValue("config/dc_timeout", dcPage->txtTimeout->text().toLong());
	 settings->setValue("config/dc_cpu_keyspace_pct", dcPage->txtCPUTokenPercentage->text().toLong()); 
	

	 QStringList lst;
	 
	 for(int i=0; i < dcPage->serverList->count(); i++) {
		 QListWidgetItem *itm = dcPage->serverList->item(i);
		 lst.append(itm->text());
	 }

	 settings->setValue("config/dc_hosts", lst);
	 
	 event->accept();

	 // if error processing, we can ignore close and force user to fix
	 // event->ignore();
 }