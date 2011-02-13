#ifndef __BL4CKJACK_DISTRIBUTED_SERVER_H__
#define __BL4CKJACK_DISTRIBUTED_SERVER_H__

#include "bl4ckJack_distributed_service.h"

#include <QThread>
#include <RCF/ClientProgress.hpp>

#ifndef DEFAULT_PORT
#define DEFAULT_PORT	40201
#endif

//! DistributedServer Class
/**
 * DistributedServer Class
 * DistributedServer Class used for listening for new keyspaces and computing as necessary.
 */
class DistributedServer : public QThread
{
	Q_OBJECT

public:
	//! DistributedServer constructor
	/**
	  * DistributedServer constructor
	  * Used for listening for new keyspaces and computing as necessary.
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	DistributedServer() {
		this->port = DEFAULT_PORT;
		strncpy(this->localHost, "127.0.0.1", sizeof(this->localHost) - 1);
	}

	//! DistributedServer Deconstructor
	/**
	  * DistributedServer deconstructor
	  * Used for listening for new keyspaces and computing as necessary.
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	~DistributedServer();

	
	//! DistributedServer startProgress
	/**
	  * DistributedServer startProgress
	  * Begin the progress gathering process.
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	  
	void startProgress();
	
	//! DistributedServer setArgs
	/**
	  * DistributedServer setArgs
	  * Set argument combination of server and port for listening.
	  * @param character pointer server
	  * @param int port
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	void setArgs(char *server, int port);
	
	//! DistributedServer setArgs
	/**
	  * DistributedServer setArgs
	  * Set argument combination of server and port for listening.
	  * @param character pointer server
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	void setArgs(char *server);
	
	//! DistributedServer setArgs
	/**
	  * DistributedServer setArgs
	  * Set argument combination of server and port for listening.
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
	void setArgs();
	
	//! DistributedServer run
	/**
	  * DistributedServer run
	  * Run our distributed server.
      * @see DistributedServer()
      * @see ~DistributedServer()
      * @return None
	  */
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