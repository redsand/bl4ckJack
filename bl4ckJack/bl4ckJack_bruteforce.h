#ifndef __BL4CKJACK_BRUTEFORCE_H__
#define __BL4CKJACK_BRUTEFORCE_H__

#include <Qt>
#include <QMutex>

#include "bl4ckJack_btree.h"
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

class BruteForceMatch {
public:
	std::string hash;
	std::string password;

	void serialize(SF::Archive &archive) {
		archive & hash & password;
	}
};

class BruteForceStats {
public:
		BruteForceStats() { }
		void setHashSec(float m) {
			this->milHashSec = m;
		}
		void setHashFound(float m) {
			this->totalHashFound = m;
		}
		void setCurrentTokens(long double start, long double stop) {
			this->currentStartToken = start;
			this->currentStopToken = stop;
		}

		void serialize(SF::Archive &archive) {
			archive & milHashSec & totalHashFound & currentStartToken & currentStopToken;
		}

// private:
	float milHashSec; // hash/sec in millions
	float totalHashFound;
	long double currentStartToken;
	long double currentStopToken;

};


class BruteForce {

public:

	BruteForce() {
		this->base = NULL;
		this->stats.milHashSec = this->stats.currentStartToken = this->stats.currentStopToken = this->stats.totalHashFound = 0;
	}

	~BruteForce() {
		stop();
		if(this->base) {
			delete this->base;
			this->base = NULL;
		}
	}
	
	BruteForceStats getStats(BruteForceStats &s) {
		
		this->statsMutex.lock();
		s.milHashSec = this->stats.milHashSec;
		s.totalHashFound = this->stats.totalHashFound;
		s.currentStartToken = this->stats.currentStartToken;
		s.currentStopToken = this->stats.currentStopToken;
		this->statsMutex.unlock();
		return s;
	}

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

	void setCharset(char *set) {
		if(this->base)
			delete this->base;

		this->base = new BaseConversion(set);
	}

	BinSTree *getBTree(void) {
		return & this->btree;
	}

	void setModule(std::string arg) {
		this->EnabledModule = arg;
	}

	void addBTree(void *hash, size_t s);
	void delBTree(void *hash, size_t s);
	bool findBTree(void *hash, size_t s);

	void start(std::vector< std::pair< long double, long double> > *keyspaceList);
	void stop();


#ifdef WIN32
	static unsigned long  _stdcall NewThread(void *);
	unsigned long NewThread();

	static unsigned long  _stdcall NewThreadCPU(void *);
	unsigned long NewThreadCPU();

	static unsigned long  _stdcall NewThreadGPU(void *);
	unsigned long NewThreadGPU();

#else
	void *NewThread(void *);
	void *NewThread();
#endif

private:
	
	BaseConversion *base;
	
	BruteForceStats stats;
	QMutex statsMutex;

	BinSTree btree;

	std::vector< BruteForceMatch > matchList;
	
	std::vector< std::pair< long double, long double> > *keyspaceList;
	std::vector< std::pair< long double, long double> > GPUkeyspaceList;
	std::vector< std::pair< long double, long double> > CPUkeyspaceList;

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