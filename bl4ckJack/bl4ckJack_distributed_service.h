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

//! RemoteServiceImpl Class
/**
 * RemoteServiceImpl Class
 * RemoteServiceImpl Class used for binding and openining up functions for public use.
 */
class RemoteServiceImpl
{
public:

	//! RemoteServiceImpl constructor
	/**
	  * RemoteServiceImpl constructor
	  * Used for binding and openining up functions for public use.
      * @see RemoteServiceImpl()
      * @see ~RemoteServiceImpl()
      * @return None
	  */
	RemoteServiceImpl() {
		this->brute = new BruteForce();
	}

	//! RemoteServiceImpl Deconstructor
	/**
	  * RemoteServiceImpl deconstructor
	  * Used for binding and openining up functions for public use.
      * @see RemoteServiceImpl()
      * @see ~RemoteServiceImpl()
      * @return None
	  */
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

	//! RemoteServiceImpl Get Statistics 
	/**
	  * RemoteServiceImpl Get Statistics
	  * Get statistics of specific node
	  * @param BruteForceStats &
      * @see class BruteForceStats
      * @return BruteForceStats
	  */
	BruteForceStats getStats(BruteForceStats &s) {
		return this->brute->getStats(s);
	}
	
	//! RemoteServiceImpl Get Matches 
	/**
	  * RemoteServiceImpl Get Matches
	  * Get matches of specific node
	  * @param BruteForceMatch &
      * @see class BruteForceMatch
      * @return BruteForceMatch
	  */
	BruteForceMatch getMatch(BruteForceMatch &s) {
		return this->brute->getMatch(s);
	}

	//! Clear Keyspace
	/**
	  * Clear Keyspace
	  * Clear keyspace from current list on this node.
      * @see initKeyspace()
      * @return void
	  */
	void clearKeyspace(void);
	
	//! Initiate Hash
	/**
	  * Initiate Hash
	  * Initiate Hash on current list.
      * @return void
	  */
	void initHash(void);
	
	//! Initiate Keyspace
	/**
	  * Initiate Keyspace
	  * Initiate keyspace on current list.
      * @see clearKeyspace()
	  * @see submitKeyspace()
      * @return void
	  */
	void initKeyspace(void);
	
	//! Initiate Module
	/**
	  * Initiate Module
	  * Initiate module to be used on this node.
      * @return void
	  */
	void initModule(void);
	
	//! Start Distributed Service
	/**
	  * Start Distributed Service
	  * Start distributed service on this node.
      * @see stop()
      * @return void
	  */
	void start();
	
	//! Stop Distributed Service
	/**
	  * Stop Distributed Service
	  * Stop distributed service on this node.
      * @see start()
      * @return void
	  */
	void stop();
	
	//! Submit Keyspace
	/**
	  * Submit Keyspace
	  * Submit keyspace to current list on this node.
	  * @param long double startPermutation
	  * @param long double stopPermutation
      * @see initKeyspace()
	  * @see clearKeyspace()
      * @return BOOL
	  */
	BOOL submitKeyspace(long double, long double);
	

private:

	BruteForce *brute;
	
	std::string charset;
	std::string EnabledModule;
	std::vector< std::string > hashList;

};

#endif