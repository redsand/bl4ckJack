#ifndef __BL4CKJACK_BRUTEFORCE_H__
#define __BL4CKJACK_BRUTEFORCE_H__

#include <Qt>
#include <QMutex>
#include <QDebug>
#include <QString>

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

	void start();
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

	
	std::list< std::pair< long double, long double> > keyspaceList;

private:
	
	BaseConversion *base;
	
	BruteForceStats stats;
	QMutex statsMutex;

	BinSTree btree;

	BOOL stopRunning;

	int cpuCount;
	BOOL hyperThreading;
	std::vector <int> cpuThreads;
	std::vector <long double> cpuCurrent;
	int gpuCount;
	std::vector <int> gpuThreads;
	std::vector <long double> gpuCurrent;

	std::vector< BruteForceMatch > matchList;
	
	std::list <std::pair< long double, long double> > CPUkeyspaceList;
	std::list <std::pair< long double, long double> > GPUkeyspaceList;

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