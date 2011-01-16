/********************************************************************************
** Form generated from reading UI file 'bl4ckjack.ui'
**
** Created: Wed Jan 5 20:40:49 2011
**      by: Qt User Interface Compiler version 4.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BL4CKJACK_H
#define UI_BL4CKJACK_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QProgressBar>
#include <QtGui/QTableView>
#include <QtGui/QTableWidget>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_bl4ckJackClass
{
public:
    QAction *action_About;
    QAction *action_Updates;
    QAction *action_New;
    QAction *action_Open;
    QAction *action_Save;
    QAction *actionSave_As;
    QAction *action_Close;
    QAction *action_Quit;
    QAction *action_Properties;
    QAction *action_Start;
    QAction *action_Pause;
    QAction *actionS_top;
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QTableView *tblHash;
    QProgressBar *progressBar;
    QLabel *lblTotalHashestxt;
    QLabel *lblPPStxt;
    QLabel *lblCompletionTimetxt;
    QLabel *lblRecoveredPasswordstxt;
    QGroupBox *groupBox_2;
    QTableWidget *tblPassword;
    QLabel *lblRecoveredPasswords;
    QLabel *lblCompletionTime;
    QLabel *lblPPS;
    QLabel *lblTotalHashes;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Help;
    QMenu *menu_Control;

    void setupUi(QMainWindow *bl4ckJackClass)
    {
        if (bl4ckJackClass->objectName().isEmpty())
            bl4ckJackClass->setObjectName(QString::fromUtf8("bl4ckJackClass"));
        bl4ckJackClass->resize(611, 476);
        bl4ckJackClass->setMinimumSize(QSize(611, 476));
        bl4ckJackClass->setMaximumSize(QSize(611, 476));
        QIcon icon;
        icon.addFile(QString::fromUtf8("../bl4ckJack.gif"), QSize(), QIcon::Normal, QIcon::Off);
        bl4ckJackClass->setWindowIcon(icon);
        action_About = new QAction(bl4ckJackClass);
        action_About->setObjectName(QString::fromUtf8("action_About"));
        action_Updates = new QAction(bl4ckJackClass);
        action_Updates->setObjectName(QString::fromUtf8("action_Updates"));
        action_New = new QAction(bl4ckJackClass);
        action_New->setObjectName(QString::fromUtf8("action_New"));
        action_Open = new QAction(bl4ckJackClass);
        action_Open->setObjectName(QString::fromUtf8("action_Open"));
        action_Save = new QAction(bl4ckJackClass);
        action_Save->setObjectName(QString::fromUtf8("action_Save"));
        actionSave_As = new QAction(bl4ckJackClass);
        actionSave_As->setObjectName(QString::fromUtf8("actionSave_As"));
        action_Close = new QAction(bl4ckJackClass);
        action_Close->setObjectName(QString::fromUtf8("action_Close"));
        action_Quit = new QAction(bl4ckJackClass);
        action_Quit->setObjectName(QString::fromUtf8("action_Quit"));
        action_Properties = new QAction(bl4ckJackClass);
        action_Properties->setObjectName(QString::fromUtf8("action_Properties"));
        action_Start = new QAction(bl4ckJackClass);
        action_Start->setObjectName(QString::fromUtf8("action_Start"));
        action_Pause = new QAction(bl4ckJackClass);
        action_Pause->setObjectName(QString::fromUtf8("action_Pause"));
        actionS_top = new QAction(bl4ckJackClass);
        actionS_top->setObjectName(QString::fromUtf8("actionS_top"));
        centralWidget = new QWidget(bl4ckJackClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 591, 161));
        tblHash = new QTableView(groupBox);
        tblHash->setObjectName(QString::fromUtf8("tblHash"));
        tblHash->setGeometry(QRect(10, 20, 571, 131));
        progressBar = new QProgressBar(centralWidget);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setGeometry(QRect(10, 400, 581, 23));
        progressBar->setValue(0);
        lblTotalHashestxt = new QLabel(centralWidget);
        lblTotalHashestxt->setObjectName(QString::fromUtf8("lblTotalHashestxt"));
        lblTotalHashestxt->setGeometry(QRect(20, 370, 141, 16));
        lblPPStxt = new QLabel(centralWidget);
        lblPPStxt->setObjectName(QString::fromUtf8("lblPPStxt"));
        lblPPStxt->setGeometry(QRect(20, 350, 141, 16));
        lblCompletionTimetxt = new QLabel(centralWidget);
        lblCompletionTimetxt->setObjectName(QString::fromUtf8("lblCompletionTimetxt"));
        lblCompletionTimetxt->setGeometry(QRect(310, 350, 141, 16));
        lblRecoveredPasswordstxt = new QLabel(centralWidget);
        lblRecoveredPasswordstxt->setObjectName(QString::fromUtf8("lblRecoveredPasswordstxt"));
        lblRecoveredPasswordstxt->setGeometry(QRect(310, 370, 141, 16));
        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 180, 591, 151));
        tblPassword = new QTableWidget(groupBox_2);
        if (tblPassword->columnCount() < 3)
            tblPassword->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tblPassword->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        tblPassword->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        tblPassword->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        tblPassword->setObjectName(QString::fromUtf8("tblPassword"));
        tblPassword->setGeometry(QRect(10, 20, 571, 121));
        lblRecoveredPasswords = new QLabel(centralWidget);
        lblRecoveredPasswords->setObjectName(QString::fromUtf8("lblRecoveredPasswords"));
        lblRecoveredPasswords->setGeometry(QRect(460, 370, 121, 16));
        lblRecoveredPasswords->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        lblCompletionTime = new QLabel(centralWidget);
        lblCompletionTime->setObjectName(QString::fromUtf8("lblCompletionTime"));
        lblCompletionTime->setGeometry(QRect(460, 350, 121, 16));
        lblCompletionTime->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        lblPPS = new QLabel(centralWidget);
        lblPPS->setObjectName(QString::fromUtf8("lblPPS"));
        lblPPS->setGeometry(QRect(170, 350, 121, 16));
        lblPPS->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        lblTotalHashes = new QLabel(centralWidget);
        lblTotalHashes->setObjectName(QString::fromUtf8("lblTotalHashes"));
        lblTotalHashes->setGeometry(QRect(170, 370, 121, 16));
        lblTotalHashes->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        bl4ckJackClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(bl4ckJackClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 611, 21));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QString::fromUtf8("menu_File"));
        menu_Help = new QMenu(menuBar);
        menu_Help->setObjectName(QString::fromUtf8("menu_Help"));
        menu_Control = new QMenu(menuBar);
        menu_Control->setObjectName(QString::fromUtf8("menu_Control"));
        bl4ckJackClass->setMenuBar(menuBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_Control->menuAction());
        menuBar->addAction(menu_Help->menuAction());
        menu_File->addAction(action_New);
        menu_File->addAction(action_Open);
        menu_File->addSeparator();
        menu_File->addAction(action_Save);
        menu_File->addAction(actionSave_As);
        menu_File->addAction(action_Close);
        menu_File->addSeparator();
        menu_File->addAction(action_Properties);
        menu_File->addSeparator();
        menu_File->addAction(action_Quit);
        menu_Help->addAction(action_About);
        menu_Help->addSeparator();
        menu_Help->addAction(action_Updates);
        menu_Control->addAction(action_Start);
        menu_Control->addSeparator();
        menu_Control->addAction(action_Pause);
        menu_Control->addSeparator();
        menu_Control->addAction(actionS_top);

        retranslateUi(bl4ckJackClass);

        QMetaObject::connectSlotsByName(bl4ckJackClass);
    } // setupUi

    void retranslateUi(QMainWindow *bl4ckJackClass)
    {
        bl4ckJackClass->setWindowTitle(QApplication::translate("bl4ckJackClass", "bl4ckJack", 0, QApplication::UnicodeUTF8));
        action_About->setText(QApplication::translate("bl4ckJackClass", "&About", 0, QApplication::UnicodeUTF8));
        action_Updates->setText(QApplication::translate("bl4ckJackClass", "&Updates", 0, QApplication::UnicodeUTF8));
        action_New->setText(QApplication::translate("bl4ckJackClass", "&New", 0, QApplication::UnicodeUTF8));
        action_Open->setText(QApplication::translate("bl4ckJackClass", "&Open", 0, QApplication::UnicodeUTF8));
        action_Save->setText(QApplication::translate("bl4ckJackClass", "&Save", 0, QApplication::UnicodeUTF8));
        actionSave_As->setText(QApplication::translate("bl4ckJackClass", "Save &As", 0, QApplication::UnicodeUTF8));
        action_Close->setText(QApplication::translate("bl4ckJackClass", "&Close", 0, QApplication::UnicodeUTF8));
        action_Quit->setText(QApplication::translate("bl4ckJackClass", "&Quit", 0, QApplication::UnicodeUTF8));
        action_Properties->setText(QApplication::translate("bl4ckJackClass", "&Properties", 0, QApplication::UnicodeUTF8));
        action_Start->setText(QApplication::translate("bl4ckJackClass", "&Start", 0, QApplication::UnicodeUTF8));
        action_Pause->setText(QApplication::translate("bl4ckJackClass", "&Pause", 0, QApplication::UnicodeUTF8));
        actionS_top->setText(QApplication::translate("bl4ckJackClass", "S&top", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("bl4ckJackClass", "Password Work Bench", 0, QApplication::UnicodeUTF8));
        lblTotalHashestxt->setText(QApplication::translate("bl4ckJackClass", "Total Uncracked Hashes:", 0, QApplication::UnicodeUTF8));
        lblPPStxt->setText(QApplication::translate("bl4ckJackClass", "Passwords Per Second (PPS) ", 0, QApplication::UnicodeUTF8));
        lblCompletionTimetxt->setText(QApplication::translate("bl4ckJackClass", "Time Until Completion: ", 0, QApplication::UnicodeUTF8));
        lblRecoveredPasswordstxt->setText(QApplication::translate("bl4ckJackClass", "Total Recovered Passwords: ", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("bl4ckJackClass", "Recovered Passwords", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem = tblPassword->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("bl4ckJackClass", "Module", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = tblPassword->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("bl4ckJackClass", "Password", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = tblPassword->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("bl4ckJackClass", "Hash", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        tblPassword->setToolTip(QApplication::translate("bl4ckJackClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" width: 60px; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Right click for more options.</p></body></html>", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        lblRecoveredPasswords->setText(QString());
        lblCompletionTime->setText(QString());
        lblPPS->setText(QString());
        lblTotalHashes->setText(QString());
        menu_File->setTitle(QApplication::translate("bl4ckJackClass", "&File", 0, QApplication::UnicodeUTF8));
        menu_Help->setTitle(QApplication::translate("bl4ckJackClass", "&Help", 0, QApplication::UnicodeUTF8));
        menu_Control->setTitle(QApplication::translate("bl4ckJackClass", "&Control", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class bl4ckJackClass: public Ui_bl4ckJackClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BL4CKJACK_H
