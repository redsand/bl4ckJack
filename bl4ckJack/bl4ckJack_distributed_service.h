#ifndef __BL4CKJACK_DISTRIBUTED_SERVICE_H__
#define __BL4CKJACK_DISTRIBUTED_SERVICE_H__

#include <string>
#include <vector>

#include <RCF/Idl.hpp>
#include <SF/vector.hpp>
#include <SF/memory.hpp> 

#include <Qt>
#include <QThread>
#include "bl4ckJack_bruteforce.h"

RCF_BEGIN(RemoteService, "RemoteService")
	RCF_METHOD_R1(BruteForceStats, getStats, BruteForceStats &);
	RCF_METHOD_R1(BruteForceMatch, getMatch, BruteForceMatch &);
    RCF_METHOD_R2(BOOL, submitKeyspace, long double, long double);
	RCF_METHOD_V0(void, initKeyspace);
	RCF_METHOD_V0(void, initHash);
	RCF_METHOD_V0(void, initModule);
	RCF_METHOD_V0(void, start);
	RCF_METHOD_V0(void, stop);
	RCF_METHOD_V0(void, clearKeyspace);
RCF_END(RemoteService);

class RemoteServiceImpl
{
public:

	RemoteServiceImpl() {
		this->brute = new BruteForce();
	}

	~RemoteServiceImpl() {
		try {
			delete this->brute;
		} catch(std::exception const& e) {
		}
	}


	/*
		- add keyspace to array/queue for processing
		- start computing (background batch)
		- stop computing (end background)
		- clear array/queue of keyspace
	*/

	// must be freed
	BruteForceStats getStats(BruteForceStats &s) {
		return this->brute->getStats(s);
	}
	
	BruteForceMatch getMatch(BruteForceMatch &s) {
		return this->brute->getMatch(s);
	}

	void clearKeyspace(void);
	void initHash(void);
	void initKeyspace(void);
	void initModule(void);
	void start();
	void stop();
	BOOL submitKeyspace(long double, long double);
	
private:

	BruteForce *brute;
	
	std::string charset;
	std::string EnabledModule;
	std::vector< std::string > hashList;

};

#endif