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

//! bl4ckJackBrute Class
/**
 * bl4ckJackBrute Class
 * bl4ckJackBrute Class used for managing bruteforce node
 */
class bl4ckJackBrute : public QThread {

	 Q_OBJECT

public:

	//! bl4ckJackBrute constructor
	/**
	  * bl4ckJackBrute constructor
	  * Used for managing bruteforce node
	  * @param parent widget pointer
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	bl4ckJackBrute(QObject *parent) : QThread(parent) {
		go=true;
		stop=false;
		this->listenHost = "127.0.0.1";
		this->distributedServer = NULL;
	}
	
	//! bl4ckJackBrute Deconstructor
	/**
	  * bl4ckJackBrute Deconstructor
	  * Used for managing bruteforce node
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	~bl4ckJackBrute() {
		stop=true;
		go = true;
		msleep(5000);
		if(this->distributedServer) {
			this->distributedServer->terminate();
			delete this->distributedServer;
		}
	}

	void msleep(unsigned long x) {
		return QThread::msleep(x);
	}

	//! Pause bruteforce within node.
	/**
	  * Pause bruteforce within node.
	  * Pause bruteforce within node.
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	void pause(void) {
		go = false;
	}

	//! Set Listen Host
	/**
	  * Set Listen Host
	  * Set listen source host for connections
	  * @param QString arg
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	void setListenHost(QString arg) {
		this->listenHost = arg;
	}

	//! Set Listen Only
	/**
	  * Set Listen Only
	  * Only listen on source host for connections
	  * @param bool arg
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	void setListenOnly(bool arg) {
		this->listenOnly = arg;
	}

	//! Set Module
	/**
	  * Set Module
	  * Set module used for brute forcing.
	  * @param QString arg
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return None
	  */
	void setModule(QString arg) {
		this->EnabledModule = arg;
	}
	
	//! Get Module
	/**
	  * Get Module
	  * Get module used for brute forcing.
      * @see bl4ckJackBrute()
      * @see ~bl4ckJackBrute()
      * @return QString
	  */
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
	//! If not go, then lets pause
	bool go;
	//! If true, then stop
	bool stop;
	//! If true, only listen, don't connect out to other hosts.
	bool listenOnly;
	
	//! Host to listen on
	QString listenHost;
	
	//! DistributedServer object
	DistributedServer *distributedServer;
	
	//! Internal module used for generation
	QString EnabledModule;
	
 };
 
#endif