
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <RCF/ClientProgress.hpp>
#include <RCF/Idl.hpp>
#include <RCF/IpServerTransport.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/FilterService.hpp>
#include <RCF/ZlibCompressionFilter.hpp>
#include <RCF/OpenSslEncryptionFilter.hpp>

#include <bl4ckJack.h>
#include <bl4ckJack_distributed_server.h>
#include <bl4ckJack_distributed_service.h>


DistributedServer::~DistributedServer() {
	
	//this->server->unbind<RemoteService>("RemoteService");
	this->server->stop();
	// sleep?
	try {
		delete this->server;
	} catch(...) {
	}
}

void DistributedServer::setArgs() {
	this->port = DEFAULT_PORT;
	strncpy(this->localHost, "127.0.0.1", sizeof(this->localHost) - 1);
}

void DistributedServer::setArgs(char *host) {
	this->port = DEFAULT_PORT;
	strncpy(this->localHost, host, sizeof(this->localHost) - 1);
}

void DistributedServer::setArgs(char *host, int port) {
	this->port = port;
	strncpy(this->localHost, host, sizeof(this->localHost) - 1);
}

void DistributedServer::doWork() {
	int filterIter = 0;
	this->server = new RCF::RcfServer( RCF::TcpEndpoint(this->localHost, this->port) );

	RCF::FilterServicePtr filterServicePtr( new RCF::FilterService());
	if(settings->value("config/dc_compression", true).toBool()) {
		try {
			filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr(new RCF::ZlibStatefulCompressionFilterFactory()));
			filterIter++;
		} catch(const RCF::Exception &e) {
			qDebug() << e.getErrorString().c_str();
		}
	}

	if(settings->value("config/dc_ssl_encryption", true).toBool()) {
		try {
			filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( 
				new RCF::OpenSslEncryptionFilterFactory(
					settings->value("config/dc_ssl_encryption_pem_file","path/to/serverCert.pem").toByteArray().constData(), 
					settings->value("config/dc_ssl_encryption_pem_password","password").toByteArray().constData())));
			filterIter++;
		} catch(const RCF::Exception &e) {
			qDebug() << e.getErrorString().c_str();
		}
	}

	if(filterIter > 0) {
		this->server->addService(filterServicePtr);
	}

	try {
		this->server->bind<RemoteService>(remoteService);
	} catch(const RCF::Exception &e) {
		qDebug() << e.getErrorString().c_str();
	}

	try {
		this->server->start();
	} catch(const RCF::Exception &e) {
		qDebug() << e.getErrorString().c_str();
	}

	this->server->waitForStopEvent();

}