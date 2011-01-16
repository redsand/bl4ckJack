#ifndef __BL4CKJACK_DISTRIBUTED_SERVER_H__
#define __BL4CKJACK_DISTRIBUTED_SERVER_H__

#include "bl4ckJack_distributed_service.h"

#include <QThread>
#include <RCF/ClientProgress.hpp>

#ifndef DEFAULT_PORT
#define DEFAULT_PORT	40201
#endif

class DistributedServer : public QThread
{
	Q_OBJECT

public:
	DistributedServer() {
		this->port = DEFAULT_PORT;
		strncpy(this->localHost, "127.0.0.1", sizeof(this->localHost) - 1);
	}

	~DistributedServer();

	void startProgress();
	void setArgs(char *server, int port);
	void setArgs(char *server);
	void setArgs();
	
	virtual void run() { 
		doWork();
    }
	
public slots:
	void doWork();

private:
	char localHost[1024];
	int port;
	RemoteServiceImpl remoteService;
	RCF::RcfServer *server;
};

#endif