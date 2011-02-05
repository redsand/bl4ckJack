#ifndef __BL4CKJACK_DISTRIBUTED_H__
#define __BL4CKJACK_DISTRIBUTED_H__

#pragma once

// Client == GPU/CPU bruteforcing interface
// Server == GUI asking for updates and distributing pieces of keyspace
//				keyspace: { keyspace#: 1-32, token#: ( (charsetLen ^ keyspace) / (clients * token_factor) ),
//							double startToken, double endToken
//						 }

#include <math.h>
#include <Qt>
#include <QThread>

#include "bl4ckJack_distributed_server.h"


 class bl4ckJackBrute : public QThread {

	 Q_OBJECT

public:

	bl4ckJackBrute(QObject *parent) : QThread(parent) {
		go=true;
		this->listenHost = "127.0.0.1";
		this->distributedServer = NULL;
	}
	
	~bl4ckJackBrute() {
		
		if(this->distributedServer) {
			this->distributedServer->terminate();
			delete this->distributedServer;
		}
	}

	void msleep(unsigned long x) {
		return QThread::msleep(x);
	}

	void pause(void) {
		go = false;
	}

	
	void setListenHost(QString arg) {
		this->listenHost = arg;
	}

	void setListenOnly(bool arg) {
		this->listenOnly = arg;
	}

	void setModule(QString arg) {
		this->EnabledModule = arg;
	}
	
	QString getModule(void) {
		return this->EnabledModule;
	}

	virtual void run() { doWork(); }

signals:
	void updateBruteStatus(int, int, QString);
	void updateBruteLabels(double, QString, qint64);
	void updateBruteDateTime(QString);
	void updateBrutePassword(QString, QString);

public slots:
    void doWork();

private:
	bool go;
	bool listenOnly;
	QString listenHost;
	DistributedServer *distributedServer;
	QString EnabledModule;
 };
 
#endif