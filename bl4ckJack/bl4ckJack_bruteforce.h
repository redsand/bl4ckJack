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
		BruteForceStats() { 
			milHashSec = 0;
			totalHashFound = 0;
			currentOpenTokens = 0;
		}
		void setHashSec(float m) {
			this->milHashSec = m;
		}
		void setHashFound(float m) {
			this->totalHashFound = m;
		}
		void setCurrentOpenTokens(long double s) {
			currentOpenTokens = s;
		}

		void serialize(SF::Archive &archive) {
			archive & milHashSec & totalHashFound & currentOpenTokens;
		}

// private:
	float milHashSec; // hash/sec in millions
	float totalHashFound;
	long double currentOpenTokens;
};


class BruteForce {

public:

	bool getKeyspace;

	BruteForce() {

		
		memset(&hThread, 0, sizeof(hThread));
		memset(&hThreadCPU, 0, sizeof(hThreadCPU));
		memset(&hThreadGPU, 0, sizeof(hThreadGPU));
		this->stopRunning = FALSE;
		this->getKeyspace = true;
		this->EnabledModule = "Default Module";
		this->base = NULL;
		this->stats.milHashSec = this->stats.totalHashFound = 0;
	}

	~BruteForce() {
		stop();
		if(this->base) {
			delete this->base;
			this->base = NULL;
			/*
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
			*/
		}

		// clear our btree
		this->btree.destroy();
	}
	
	BruteForceStats getStats(BruteForceStats &s) {

		this->statsMutex.lock();
		s.milHashSec = this->stats.milHashSec;
		s.totalHashFound = this->stats.totalHashFound;
		this->stats.totalHashFound = 0;
		this->stats.milHashSec = 0;
		// calc our current open tokens:
		//
		this->stats.currentOpenTokens = 0;
		for(int i = 0; i < CPUkeyspaceList.size(); i++) {
			this->stats.currentOpenTokens += (CPUkeyspaceList[i].second - CPUkeyspaceList[i].first);
		}
		s.currentOpenTokens = this->stats.currentOpenTokens;
		
		/*
		for(int i = 0; i < GPUkeyspaceList.size(); i++) {
			this->stats.currentOpenTokens += (GPUkeyspaceList[i].second - GPUkeyspaceList[i].first);
		}
		*/
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

	BOOL stopRunning;

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