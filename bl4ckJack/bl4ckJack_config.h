#ifndef BL4CKJACK_CONFIG_H
#define BL4CKJACK_CONFIG_H

#include <QWidget>
#include <QDialog>
#include <QCheckbox>
#include <QTextEdit>

#include "bl4ckJack.h"

 class ConfigurationPage : public QWidget
 {
	 Q_OBJECT

 public:
     ConfigurationPage(QWidget *parent = 0);
     ~ConfigurationPage();
	 
	 QCheckBox *hideWindowCheckBox;
	 QCheckBox *closeWindowCheckBox;
	 QCheckBox *close2WindowCheckBox;
	 QCheckBox *notifySaveFileOverwriteCheckBox;
	 QCheckBox *autoWindowCheckBox;
	 QPlainTextEdit *charsetText;
	 
 private slots:
		 void commitSysTrayData(bool);
		 void commitBackupData(bool);
		 void pushOpenCharset();
 };

 class GPUPage : public QWidget
 {
	 Q_OBJECT

 public:
     GPUPage(QWidget *parent = 0);
	 ~GPUPage() {
		 delete inputRefreshRate;
		 delete inputMaxMemInit;
		 delete inputGPUBlocks;
		 delete inputGPUThreads;
		 delete inputEnableHardwareMonitoring;
		 delete gpuList;
	 }

	QLineEdit *inputRefreshRate;
	QLineEdit *inputMaxMemInit;
	QLineEdit *inputGPUBlocks;
	QLineEdit *inputGPUThreads;
	QCheckBox *inputEnableHardwareMonitoring;
	QListWidget *gpuList;
	 
 };

 class ModulesPage : public QWidget
 {
	 Q_OBJECT

 public:
     ModulesPage(QWidget *parent = 0);

 private slots:
 };

 class DCPage : public QWidget
 {
	 Q_OBJECT

 private slots:
	 void actionAddServerFn(void);
	 void actionDelServerFn(void);
	 void customContextMenu(const QPoint &pos);

 public:
     DCPage(QWidget *parent = 0);
     ~DCPage();

	 QAction *actionAddServer;
	 QAction *actionDelServer;
	 
	 QCheckBox *enableLocalServerCheckBox;
	 QCheckBox *enableSSLCheckBox;
	 QCheckBox *enableCompressionCheckBox;
	 
	 QLineEdit *inputMaxPasswordSize;
	 QLineEdit *txtMinimumTokens;
	 QLineEdit *txtCPUTokenPercentage;
	 QLineEdit *txtTimeout;
	 QListWidget *serverList;

 private:
	 QMenu *serverListMenu;
	 
 };

 class QListWidget;
 class QListWidgetItem;
 class QStackedWidget;

 class ConfigDialog : public QDialog
 {
     Q_OBJECT

 public:
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