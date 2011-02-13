#ifndef BL4CKJACK_CONFIG_H
#define BL4CKJACK_CONFIG_H

#include <QWidget>
#include <QDialog>
#include <QCheckbox>
#include <QTextEdit>

#include "bl4ckJack.h"

//! ConfigurationPage Class
/**
 * ConfigurationPage Class
 * ConfigurationPage Class used for configuring bl4ckJack.
 */
 class ConfigurationPage : public QWidget
 {
	 Q_OBJECT

 public:
	//! ConfigurationPage constructor
	/**
	  * ConfigurationPage constructor
	  * Page to show local GUI settings relative to user experience.
	  * @param parent widget pointer
      * @see ConfigurationPage()
      * @see ~ConfigurationPage()
      * @return None
	  */
     ConfigurationPage(QWidget *parent = 0);
	 
	 //! ConfigurationPage Deconstructor
	/**
	  * ConfigurationPage Deconstructor
	  * Page to show local GUI settings relative to user experience.
	  * @param parent widget pointer
      * @see ConfigurationPage()
      * @see ~ConfigurationPage()
      * @return None
	  */
     ~ConfigurationPage();
	 
	 //! Hide Window on Close
	 QCheckBox *hideWindowCheckBox;
	 
	 //! Close Window on Close
	 QCheckBox *closeWindowCheckBox;
	 
	 //! Close Window Option #2 on Close
	 QCheckBox *close2WindowCheckBox;
	 
	 //! Notify Save File Overwrite
	 QCheckBox *notifySaveFileOverwriteCheckBox;
	 
	 //! Auto Save Status
	 QCheckBox *autoWindowCheckBox;
	 
	 //! Character Set
	 QPlainTextEdit *charsetText;
	 
 private slots:
		 void commitSysTrayData(bool);
		 void commitBackupData(bool);
		 void pushOpenCharset();
 };

//! GPUPage Class
/**
 * GPUPage Class
 * GPUPage Class used for configuring bl4ckJack.
 */
 class GPUPage : public QWidget
 {
	 Q_OBJECT

 public:
	//! GPUPage constructor
	/**
	  * GPUPage constructor
	  * Page to show local GUI settings relative to GPU options.
	  * @param parent widget pointer
      * @see GPUPage()
      * @see ~GPUPage()
      * @return None
	  */
     GPUPage(QWidget *parent = 0);
	 
	 //! GPUPage Deconstructor
	/**
	  * GPUPage Deconstructor
	  * Page to show local GUI settings relative to GPU options.
      * @see GPUPage()
      * @see ~GPUPage()
      * @return None
	  */
	 ~GPUPage() {
		 delete inputRefreshRate;
		 delete inputMaxMemInit;
		 delete inputGPUBlocks;
		 delete inputGPUThreads;
		 delete inputEnableHardwareMonitoring;
		 delete gpuList;
	 }

	 //! Input Refresh Rate
	QLineEdit *inputRefreshRate;
	
	//! Maximum Memory Limit
	QLineEdit *inputMaxMemInit;
	
	//! GPU Block Count
	QLineEdit *inputGPUBlocks;
	
	//! GPU Thread Count
	QLineEdit *inputGPUThreads;
	
	//! GPU Enable Hardware Monitoring
	QCheckBox *inputEnableHardwareMonitoring;
	
	//! GPU Device QListWidget
	QListWidget *gpuList;
	 
 };

//! ModulesPage Class
/**
 * ModulesPage Class
 * ModulesPage Class used for configuring bl4ckJack.
 */
 class ModulesPage : public QWidget
 {
	 Q_OBJECT

 public:
	//! ModulesPage constructor
	/**
	  * ModulesPage constructor
	  * Page to show local GUI settings relative to module options.
	  * @param parent widget pointer
      * @see ModulesPage()
      * @see ~ModulesPage()
      * @return None
	  */
     ModulesPage(QWidget *parent = 0);

 private slots:
 };

//! Distributed Computing Page (DCPage) Class
/**
 * Distributed Computing Page (DCPage) Class
 * Distributed Computing Page (DCPage) Class used for configuring bl4ckJack.
 */
 class DCPage : public QWidget
 {
	 Q_OBJECT

 private slots:
	 void actionAddServerFn(void);
	 void actionDelServerFn(void);
	 void customContextMenu(const QPoint &pos);

 public:
	
	//! DCPage constructor
	/**
	  * DCPage constructor
	  * Page to show local GUI settings relative to distributed computing options.
	  * @param parent widget pointer
      * @see DCPage()
      * @see ~DCPage()
      * @return None
	  */
     DCPage(QWidget *parent = 0);
	
	//! DCPage Deconstructor
	/**
	  * DCPage Deconstructor
	  * Page to show local GUI settings relative to distributed computing options.
      * @see DCPage()
      * @see ~DCPage()
      * @return None
	  */
     ~DCPage();

	 //! QAction for Adding Distributed Servers
	 QAction *actionAddServer;
	 //! QAction for Deleting Distributed Servers
	 QAction *actionDelServer;
	 
	 //! Enable Local Server Option
	 QCheckBox *enableLocalServerCheckBox;
	 //! Enable SSL Connection Option
	 QCheckBox *enableSSLCheckBox;
	 //! Enable Compression Option
	 QCheckBox *enableCompressionCheckBox;
	 
	 //! Max Password Length
	 QLineEdit *inputMaxPasswordSize;
	 
	 //! Minimum Tokens per Host
	 QLineEdit *txtMinimumTokens;
	 
	 //! Percentage of tokens for CPU
	 QLineEdit *txtCPUTokenPercentage;
	 
	 //! Timeout per host communication
	 QLineEdit *txtTimeout;
	 
	 //! QListWidget of servers available for distributed computing
	 QListWidget *serverList;

 private:
	 QMenu *serverListMenu;
	 
 };

 class QListWidget;
 class QListWidgetItem;
 class QStackedWidget;

//! ConfigDialog Page Class
/**
 * ConfigDialog Page Class
 * ConfigDialog Page Class used for configuring bl4ckJack.
 */
 class ConfigDialog : public QDialog
 {
     Q_OBJECT

 public:
	//! ConfigDialog constructor
	/**
	  * ConfigDialog constructor
	  * Page to group all other pages together
	  * @param parent widget pointer
      * @see ConfigDialog()
      * @see ~ConfigDialog()
      * @return None
	  */
     ConfigDialog(QWidget *parent = 0);

 public slots:
     void changePage(QListWidgetItem *current, QListWidgetItem *previous);

 private slots:
	void closeEvent(QCloseEvent *event);

 private:
     void createIcons();

     QListWidget *contentsWidget;
     QStackedWidget *pagesWidget;
 };

#endif