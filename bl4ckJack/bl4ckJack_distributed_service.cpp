#include <Qt>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <RCF/RcfServer.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/ZlibCompressionFilter.hpp>
#include <RCF/CurrentSession.hpp>
#include <RCF/RcfSession.hpp>
#include "bl4ckJack_distributed_service.h"
#include "bl4ckJack_distributed.h"

#include <Qt>
#include <QString>
#include <QDebug>
#include "bl4ckJack.h"

void RemoteServiceImpl::initKeyspace(void) {

	RCF::RcfSession & session = RCF::getCurrentRcfSession();
    std::string charset = session.getRequestUserData();

	if(charset.empty())
		return;

	this->charset = charset;
	qDebug() << "initKeyspace " << charset.c_str();
	settings->setValue("config/current_charset", charset.c_str());

}

void RemoteServiceImpl::initHash(void) {
	RCF::RcfSession & session = RCF::getCurrentRcfSession();
    std::string hash = session.getRequestUserData();

	if(hash.empty())
		return;

	this->hashList.push_back(hash.c_str());
	qDebug() << "initHash " << hash.c_str();
}

void RemoteServiceImpl::initModule(void) {
	RCF::RcfSession & session = RCF::getCurrentRcfSession();
    std::string module = session.getRequestUserData();

	if(module.empty())
		return;

	this->EnabledModule = module.c_str();
	qDebug() << "initModule " << module.c_str();
}

/*
	- add keyspace to array/queue for processing
	- start computing (background batch)
	- stop computing (end background)
	- clear array/queue of keyspace
 */

BOOL RemoteServiceImpl::submitKeyspace(long double start, long double stop) {

	std::pair< long double, long double > keypair;

	// if our current keypair is getting close, lets go ahead and get another.

	//this->keyspaceList.at(0).first
	//this->brute->CPUkeyspaceList.at(0).first;
	if(this->brute->getKeyspace == true) {	
		this->brute->getKeyspace = false;
		keypair.first = start;
		keypair.second = stop;
		this->brute->keyspaceList.push_back(keypair);
		qDebug() << "submitKeyspace low: " << (double)start << " high: " << (double)stop;
		return TRUE;
	} /* else
		qDebug() << "submitKeyspace request denied, low: " << (double)start << " high: " << (double)stop;
	  */
	return FALSE;
}

#define HEXTOBIN(x) ( (x) >= '0' && (x) <= '9' ? ((x)-'0') : \
                    (x) >= 'A' && (x) <= 'F' ? ((x)-'A'+10) : ((x)-'a'+10))

static LPBYTE
HexToBin (LPSTR p, int len)
{
    int i;
    LPBYTE out = (LPBYTE) malloc (len >> 1);
    LPBYTE out_org = out;

    for (i = 0; i < len; i += 2)
    {
        *out++ = (HEXTOBIN (p[i]) << 4) | HEXTOBIN (p[i + 1]);
    }
    return out_org;
}


void RemoteServiceImpl::start() {
	// start bruteforcing
	qDebug() << "start bruteforce";
	
	this->brute->setModule(this->EnabledModule);

	// start brute thread with given keyspaces
	
	qDebug() << "creating hashList b-tree in host memory.";
	// generate our BTree for comparison
	size_t slen=0;
	for(unsigned int i = (this->hashList.size() / 2); i < this->hashList.size(); i++) {
		slen = strlen(this->hashList[i].c_str());
		void *hash = (void *) HexToBin((char *)this->hashList[i].c_str(), slen);
		this->brute->addBTree(hash, slen / 2);
	}

	for(unsigned int i = 0; i < (this->hashList.size() / 2); i++) {
		slen = strlen(this->hashList[i].c_str());
		void *hash = (void *) HexToBin((char *)this->hashList[i].c_str(), slen);
		this->brute->addBTree(hash, slen / 2);
	}

	qDebug() << "hashList b-tree " << this->brute->getBTree()->getCount() << " units";

	// create our thread to manage that does all the brute forcing.
	// 

	this->brute->start();

	// needs to know our keyspace for bruteforcing

}

void RemoteServiceImpl::stop() {
	// stop bruteforcing
	qDebug() << "stop bruteforce";
	this->brute->stop();

	delete this->brute;
}

void RemoteServiceImpl::clearKeyspace(void) {
	// clear our keyspace
	qDebug() << "clearKeyspace";
	this->brute->keyspaceList.clear();
}
