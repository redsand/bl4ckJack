#ifndef __BL4CKJACK_BRUTEFORCE_H__
#define __BL4CKJACK_BRUTEFORCE_H__

#include <Qt>
#include <QMutex>
#include <QDebug>
#include <QString>

#include "bl4ckJack_base.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <SF/Archive.hpp>
#include <SF/serializer.hpp>
#include <SF/vector.hpp>
#include <SF/memory.hpp>

#define MAX_BRUTE_CHARS		64

//! BruteForce Match Entry
/**
  * BruteForce Match Entry
  */
class BruteForceMatch {
public:
	std::string hash;
	std::string password;
	
	//! BruteForceMatch Serialize
	/**
	  * BruteForceMatch Serialize
	  * Necessary for transmitting this data over the wire
	  * @param SF::Archive &
      * @see BruteForceMatch()
      * @return None
	  */

	void serialize(SF::Archive &archive) {
		archive & hash & password;
	}
};

//! BruteForceStats used for tracking stats usage
/**
  * BruteForceStats used for tracking stats usage
  */

class BruteForceStats {
public:

	//! BruteForceStats constructor
	/**
	  * BruteForceStats constructor
	  * Used for tracking stats usage
      * @see BruteForceStats()
      * @see ~BruteForceStats()
      * @return None
	  */
		BruteForceStats() { 
			milHashSec = 0;
			totalHashFound = 0;
			currentOpenTokens = 0;
		}
		
		//! Set Hashes per Second
		/**
		  * Set Hashes per Second
		  * Used to set hashes per second
		  * @param float
		  * @see BruteForceStats()
		  * @see ~BruteForceStats()
		  * @return None
		  */
		void setHashSec(float m) {
			this->milHashSec = m;
		}
		
		//! Set Hashes Found
		/**
		  * Set Hashes Found
		  * Used to set hashes found
		  * @param float
		  * @see BruteForceStats()
		  * @see ~BruteForceStats()
		  * @return None
		  */
		void setHashFound(float m) {
			this->totalHashFound = m;
		}
		
		//! Set Current Open Tokens
		/**
		  * Set Current Open Tokens
		  * Used to set current open keyspace tokens
		  * @param long double
		  * @see BruteForceStats()
		  * @see ~BruteForceStats()
		  * @return None
		  */
		void setCurrentOpenTokens(long double s) {
			currentOpenTokens = s;
		}

		//! Data Serialization
		/**
		  * Data Serialization
		  * Used to transport data over the wire
		  * @param SF::Archive
		  * @see BruteForceStats()
		  * @see ~BruteForceStats()
		  * @return None
		  */
		void serialize(SF::Archive &archive) {
			archive & milHashSec & totalHashFound & currentOpenTokens;
		}

	//! Millions of Hashes per Second
	float milHashSec; /**< hash/sec in millions */
	
	//! Total Hashes Found
	float totalHashFound; /**< total hashes found */
	
	//! Current Open Tokens
	long double currentOpenTokens; /**< current unsolved permutations */
};

//! Main BruteForce GUI Thread
/**
  * Main BruteForce GUI Thread
  */
class BruteForce {

public:

	//! If TRUE, get more keyspaces for crunching.
	bool getKeyspace;

	
	//! BruteForce Constructor
	/**
	  * BruteForce Constructor
	  * Main BruteForce thread.
	  * @see ~BruteForce()
	  * @return None
	  */
	BruteForce() {

		memset(&hThread, 0, sizeof(hThread));
		memset(&hThreadCPU, 0, sizeof(hThreadCPU));
		memset(&hThreadGPU, 0, sizeof(hThreadGPU));
		this->stopRunning = FALSE;
		this->getKeyspace = true;
		this->EnabledModule = "Default Module";
		this->base = NULL;
		this->stats.milHashSec = this->stats.totalHashFound = 0;

		hashListDefaultSize=4096;
	    hashListDefaultIter=1;
		hashListEntryLength=0;
		hashList = (unsigned char **)calloc(hashListDefaultSize * hashListDefaultIter, sizeof(unsigned char *));
		hashListLength=0;
		memset(hashList, 0, hashListDefaultSize * sizeof(unsigned char *));

	}

	//! BruteForce Deconstructor
	/**
	  * BruteForce Deconstructor
	  * Main BruteForce thread.
	  * @see BruteForce()
	  * @return None
	  */
	~BruteForce() {
		stop();
#if defined(WIN32) || defined(WIN64)
		Sleep(2500);
#else
		usleep(2500);
#endif
		this->keyspaceList.clear();
		this->CPUkeyspaceList.clear();
		this->GPUkeyspaceList.clear();
		
		if(this->base) {
			delete this->base;
			this->base = NULL;

			if(hThread) {
#ifdef WIN32
				TerminateThread(hThread, 0);
#else
				pthread_kill(hThread, 0);
#endif
			}
			if(hThreadCPU) {
#ifdef WIN32
				TerminateThread(hThreadCPU, 0);
#else
				pthread_kill(hThreadCPU, 0);
#endif
			}
			if(hThreadGPU) {
#ifdef WIN32
				TerminateThread(hThreadGPU, 0);
#else
				pthread_kill(hThreadGPU, 0);
#endif
			}
		
		}

		// clear our comparison array
		for(unsigned long i=0; i < hashListLength; i++)
			free(hashList[i]);
		free(hashList);

	}

	//! BruteForce Get Stats
	/**
	  * BruteForce Get Stats
	  * Get statistics from current node.
	  * @param BruteForceStats &
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return BruteForceStats
	  */
	BruteForceStats getStats(BruteForceStats &s) {
		this->statsMutex.lock();
		s.milHashSec = this->stats.milHashSec;
		s.totalHashFound = this->stats.totalHashFound;
		this->stats.totalHashFound = 0;
		this->stats.milHashSec = 0;
		// calc our current open tokens:
		//
		this->stats.currentOpenTokens = 0;
		std::pair<long double, long double> token;
		std::list<std::pair<long double, long double> >::iterator iter;
		for(iter = CPUkeyspaceList.begin(); iter != CPUkeyspaceList.end(); iter++) {
			token = *iter;
			this->stats.currentOpenTokens += (token.second - token.first);
			for(int mmm=0; mmm < this->cpuCount; mmm++) {
				if(cpuCurrent[mmm] > token.first && cpuCurrent[mmm] < token.second) {
					this->stats.currentOpenTokens -= (cpuCurrent[mmm] - token.first);
					break;
				}
			}
		}
		//qDebug() << "Currently " << (double)s.currentOpenTokens << " open tokens";

		for(iter = GPUkeyspaceList.begin(); iter != GPUkeyspaceList.end(); iter++) {
			token = *iter;
			this->stats.currentOpenTokens += (token.second - token.first);
			for(int mmm=0; mmm < this->cpuCount; mmm++) {
				if(gpuCurrent[mmm] > token.first && gpuCurrent[mmm] < token.second) {
					this->stats.currentOpenTokens -= (gpuCurrent[mmm] - token.first);
					break;
				}
			}
		}

		s.currentOpenTokens = this->stats.currentOpenTokens;

		this->statsMutex.unlock();
		return s;
	}

	//! BruteForce Get Match
	/**
	  * BruteForce Get Match
	  * Get matches from current node.
	  * @param BruteForceMatch &
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return BruteForceMatch
	  */
	BruteForceMatch getMatch(BruteForceMatch &s) {
		if(this->matchList.size() > 0) {
			s.hash = this->matchList.at(0).hash;
			s.password = this->matchList.at(0).password;
			this->matchList.pop_back();
		} else {
			s.hash = "";
			s.password = "";
		}
		return s;
	}

	//! BruteForce Set Charset
	/**
	  * BruteForce Set Charset
	  * Set character set used for permutations
	  * @param character pointer
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void setCharset(char *set) {
		if(this->base)
			delete this->base;

		this->base = new BaseConversion(set);
	}

	//! BruteForce Set Module
	/**
	  * BruteForce Set Module
	  * Set module used for matching permutations
	  * @param character pointer
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void setModule(std::string arg) {
		this->EnabledModule = arg;
	}


	//! BruteForce Get Hash List
	/**
	  * BruteForce Get Hash List
	  * @see getHashListCount()
	  * @see hashList
	  * @see hashListLength
	  * @return unsigned char **
	  */
	unsigned char **getHashList(void) {
		return this->hashList;
	}

	//! BruteForce Get Hash List Size
	/**
	  * BruteForce Get Hash List Size
	  * @see getHashList()
	  * @see ~BruteForce()
	  * @see hashList
	  * @see hashListLength
	  * @return unsigned char **
	  */
	unsigned long getHashListCount(void) {
		return this->hashListLength;
	}
	
	//! BruteForce Add Hash List
	/**
	  * BruteForce Add Hash List
	  * Add hash to concurrent list
	  * @param void pointer to data
	  * @param size_t length of data
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void addHash(void *hash, size_t s);
	
	//! BruteForce Delete Hash List
	/**
	  * BruteForce Delete Hash List
	  * Delete hash from the hashList
	  * @param void pointer to data
	  * @param size_t length of data
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void delHash(void *hash, size_t s);
	
	//! BruteForce Find Hash
	/**
	  * BruteForce Find Hash
	  * Find hash within the hashList
	  * @param void pointer to data
	  * @param size_t length of data
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return bool
	  */
	bool findHash(void *hash, size_t s);

	//! BruteForce Re-Order Hash
	/**
	  * BruteForce Re-Order Hash
	  * Re-order hash via an iterative merge sort algorithm
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return void
	  */
	void reorderHash();

	//! BruteForce Start
	/**
	  * BruteForce Start
	  * Start brute force thread
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void start();
	
	//! BruteForce Stop
	/**
	  * BruteForce Stop
	  * Stop brute force thread
	  * @see BruteForce()
	  * @see ~BruteForce()
	  * @return None
	  */
	void stop();


#ifdef WIN32
	static unsigned long  _stdcall NewThread(void *);
	unsigned long NewThread();

	static unsigned long  _stdcall NewThreadCPU(void *);
	unsigned long NewThreadCPU(void *param, int thread_id);

	static unsigned long  _stdcall NewThreadGPU(void *);
	unsigned long NewThreadGPU(void *param, int thread_id);

#else
	void *NewThread(void *);
	void *NewThread();
#endif

	//! Keyspace list of permutation ranges
	std::list< std::pair< long double, long double> > keyspaceList;

private:
	
	BaseConversion *base;
	
	BruteForceStats stats;
	QMutex statsMutex;

	//! Internal Hash List used for hash matching
	unsigned char **hashList;
	//! Internal Hash List Length used for hash matching
	unsigned long hashListLength;
	//! Internal Hash List default Length and growth iteration;
	unsigned int hashListDefaultSize;
	//! Internal Hash List default Lenght iterator (this * hashListDefaultSize == total size)
	unsigned int hashListDefaultIter;
	//! Internal Default Entry Length, per hash
	unsigned long hashListEntryLength;

	BOOL stopRunning;

	int cpuCount;
	BOOL hyperThreading;
	std::vector <int> cpuThreads;
	std::vector <long double> cpuCurrent;
	int gpuCount;
	std::vector <int> gpuThreads;
	std::vector <long double> gpuCurrent;

	std::vector< BruteForceMatch > matchList;
	
	//! Broken up keyspace list for CPU Bruteforcing
	std::list <std::pair< long double, long double> > CPUkeyspaceList;
	
	//! Broken up keyspace list for GPU Bruteforcing
	std::list <std::pair< long double, long double> > GPUkeyspaceList;

	//! Module Enabled for bruteforcing
	std::string EnabledModule;

#ifdef WIN32
	unsigned long threadId, threadIdCPU, threadIdGPU;
#ifndef HANDLE
#define HANDLE void *
#endif
	HANDLE hThread;
	HANDLE hThreadCPU;
	HANDLE hThreadGPU;
#undef HANDLE
#else
	pthread_t threadId;
	pthread_t threadIdCPU;
	pthread_t threadIdGPU;
#endif

};

#endif