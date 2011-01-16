
#include "bl4ckJack_bruteforce.h"
#include "bl4ckJack_distributed.h"

#include "bl4ckJack_btree.h"
#include "bl4ckJack.h"
#include "bl4ckJack_timer.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <Qt>
#include <QString>
#include <QDebug>

static char *
bintohex(unsigned int len, char *bindata);

void BruteForce::addBTree(void *hash, size_t s) {
	this->btree.insert(hash, s);
}

void BruteForce::delBTree(void *hash, size_t s) {
	this->btree.remove(hash, s);
} 

bool BruteForce::findBTree(void *hash, size_t s) {
	return this->btree.find(hash, s);
}

#ifdef WIN32
unsigned long _stdcall BruteForce::NewThread(void *param) {
#else
void *BruteForce::NewThread(void *param) {
#endif
	BruteForce *self = (BruteForce*) param;
	return self->NewThread();
}

#ifdef WIN32
unsigned long BruteForce::NewThread() {
#else
void *BruteForce::NewThread() {
#endif

	// find our keyspaces
	// our 1st keyspace:
	//		- break into host and gpu keys
	//		- start host key cracker
	//		- start gpu key cracker
	//		- get updates and send back

	//this->keyspaceList->pop_back();
	
#ifdef WIN32
	hThreadCPU = CreateThread(NULL, 0, BruteForce::NewThreadCPU, this, 0, &threadIdCPU);
#else
	pthread_create(&threadIdCPU, BruteForce::NewThreadCPU, this);
#endif

#ifdef WIN32
	hThreadGPU = CreateThread(NULL, 0, BruteForce::NewThreadGPU, this, 0, &threadIdGPU);
#else
	pthread_create(&threadIdGPU, BruteForce::NewThreadGPU, this);
#endif

	while(true) {

#ifdef WIN32
		Sleep(500);
#else
		usleep(500);
#endif

		while(!keyspaceList->empty()) {
			
			int pct = settings->value("config/dc_cpu_keyspace_pct", 10).toInt();
			if(pct <= 0) pct = 1;

			std::pair< long double, long double> pair;

			long double space = keyspaceList->at(0).second - keyspaceList->at(0).first;
			long double cpu_amnt = (long double) (space * (float)((float)pct / 100));

			// check and make sure we have gpus to use, otherwise we're going to rely strictly on CPU
			
			// per CPU single cpu then:
			// http://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
			DWORD cpuCount = 0;
#ifdef WIN32
			SYSTEM_INFO sysinfo;
			GetSystemInfo( &sysinfo );

			cpuCount = sysinfo.dwNumberOfProcessors;
#else
			cpuCount = sysconf( _SC_NPROCESSORS_ONLN );

#endif

			long double cur=keyspaceList->at(0).second - keyspaceList->at(0).first;
			while(cur > 0) {

				long double sub_amnt = (long double) (cpu_amnt / cpuCount);
				for(DWORD mine = 0; mine < cpuCount; mine++) {
					pair.first = keyspaceList->at(0).first + ((mine) * sub_amnt);
					pair.second = keyspaceList->at(0).first + ((mine) * sub_amnt) + sub_amnt;
					CPUkeyspaceList.push_back(pair);
					cur -= (pair.second - pair.first);
				}

				// per GPU
				// if gpu available {
				/*
				pair.first = keyspaceList->at(0).first + cpu_amnt;
				pair.second = keyspaceList->at(0).second;
				GPUkeyspaceList.push_back(pair);
				*/

			}

			keyspaceList->pop_back();
		}
		
	}

#ifdef WIN32
	ExitThread(0);
	return 0;
#else
	pthread_exit(0);
	return 0;
#endif

}


#ifdef WIN32
unsigned long _stdcall BruteForce::NewThreadCPU(void *param) {
#else
void *BruteForce::NewThreadCPU(void *param) {
#endif
	BruteForce *self = (BruteForce*) param;
	return self->NewThreadCPU();
}

#ifdef WIN32
unsigned long BruteForce::NewThreadCPU() {
#else
void *BruteForce::NewThreadCPU(char *charset) {
#endif

	Timer t;
	int hashFound = 0;
	unsigned char bruteStr[MAX_BRUTE_CHARS+1];
	unsigned char results[MAX_BRUTE_CHARS+1];

	std::string charset = settings->value("config/current_charset","empty charset").toString().toStdString();
	BaseConversion *base = new BaseConversion(charset);

	// this thread is responsible for bruteforcing CPU

	// identify our keyspace and iterate through our permutations (non recursively)

	this->stats.milHashSec = 0; // hash/sec in millions
	this->stats.totalHashFound = 0;
	this->stats.currentStartToken = 0;
	this->stats.currentStopToken = 0;

	while (true) {
		
		while(!CPUkeyspaceList.empty()) {
			// load our charset and keyspace and begin bruteing

			int j=0;
			for(j = 0; j < bl4ckJackModules.count(); j++) {
				if(this->EnabledModule.compare(bl4ckJackModules[j]->moduleInfo->name) == 0)
					break;
			}

			if(j >= bl4ckJackModules.count()) {
				//qDebug() << "unable to compare " << this->EnabledModule << " with any available module."
			} else {
				Timer t;
				qint64 startTime = t.StartTiming(), stopTime = 0;
				this->stats.currentStartToken = CPUkeyspaceList.at(0).first;

				for(long double i = CPUkeyspaceList.at(0).first; i < CPUkeyspaceList.at(0).second; i++) {
					base->ToBase(i, (char *)bruteStr, MAX_BRUTE_CHARS);
					size_t retLen;
					bl4ckJackModules[j]->pfbl4ckJackGenerate((unsigned char *)results, &retLen, (unsigned char *)bruteStr, strlen((const char *)bruteStr));

					// gen our hash and check for existance in hash list
					if(this->btree.find(results, retLen)) {
						hashFound++;
						BruteForceMatch match;
						char *hex = bintohex(retLen, (char *)results);
						match.hash = std::string(hex);
						match.password = std::string((char *)bruteStr);
						this->matchList.push_back(match);
						//qDebug() << "successfully broke password " << bruteStr << " with result: " << hex;
						free(hex);
					}

					if(fmod(i, 50000) == 0) {
						//qDebug() << "checking time elapse";
						if(t.ElapsedTiming(startTime, t.StopTiming()) >= 1000) {
							//qDebug() << "time elapse success " << startTime << " and " << t.StopTiming() << " is " << t.ElapsedTiming(startTime, t.StopTiming());
							this->statsMutex.lock();
							this->stats.totalHashFound += hashFound;
							hashFound = 0;

							this->stats.currentStopToken = i;
							this->stats.milHashSec = ((this->stats.currentStopToken - this->stats.currentStartToken) / 1000000 /*million*/);
							this->stats.currentStartToken = i;

							this->statsMutex.unlock();
							startTime = t.StartTiming();
						}
					}
				}
			}
			CPUkeyspaceList.pop_back();
		}

#ifdef WIN32
		Sleep(500);
#else
		usleep(500);
#endif

	}

#ifdef WIN32
	ExitThread(0);
	return 0;
#else
	pthread_exit(0);
	return 0;
#endif

}

#ifdef WIN32
unsigned long _stdcall BruteForce::NewThreadGPU(void *param) {
#else
void *BruteForce::NewThreadGPU(void *param) {
#endif
	BruteForce *self = (BruteForce*) param;
	return self->NewThreadGPU();
}

#ifdef WIN32
unsigned long BruteForce::NewThreadGPU() {
#else
void *BruteForce::NewThreadGPUGPU() {
#endif

	
	while (true) {
		
		while(!GPUkeyspaceList.empty()) {
			// load our charset and keyspace and begin bruteing

			GPUkeyspaceList.at(0).first;
			GPUkeyspaceList.at(0).second;

			GPUkeyspaceList.pop_back();
		}

#ifdef WIN32
		Sleep(500);
#else
		usleep(500);
#endif

	}


#ifdef WIN32
	ExitThread(0);
	return 0;
#else
	pthread_exit(0);
	return 0;
#endif

}

void BruteForce::start(std::vector< std::pair< long double, long double> > *keyspaceList) {

	
	this->keyspaceList = keyspaceList;
	// qDebug()

#ifdef WIN32
	hThread = CreateThread(NULL, 0, BruteForce::NewThread, this, 0, &threadId);
#else
	pthread_create(&threadId, BruteForce::NewThread, this);
#endif

}

void BruteForce::stop() {

#ifdef WIN32
	TerminateThread(this->hThread, 0);
#else
	pthread_cancel(this->threadId);
#endif

}


static char *hex_table[] = {            /* for printing hex digits */
    "00", "01", "02", "03", "04", "05", "06", "07",
    "08", "09", "0a", "0b", "0c", "0d", "0e", "0f",
    "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "1a", "1b", "1c", "1d", "1e", "1f",
    "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "2a", "2b", "2c", "2d", "2e", "2f",
    "30", "31", "32", "33", "34", "35", "36", "37",
    "38", "39", "3a", "3b", "3c", "3d", "3e", "3f",
    "40", "41", "42", "43", "44", "45", "46", "47",
    "48", "49", "4a", "4b", "4c", "4d", "4e", "4f",
    "50", "51", "52", "53", "54", "55", "56", "57",
    "58", "59", "5a", "5b", "5c", "5d", "5e", "5f",
    "60", "61", "62", "63", "64", "65", "66", "67",
    "68", "69", "6a", "6b", "6c", "6d", "6e", "6f",
    "70", "71", "72", "73", "74", "75", "76", "77",
    "78", "79", "7a", "7b", "7c", "7d", "7e", "7f",
    "80", "81", "82", "83", "84", "85", "86", "87",
    "88", "89", "8a", "8b", "8c", "8d", "8e", "8f",
    "90", "91", "92", "93", "94", "95", "96", "97",
    "98", "99", "9a", "9b", "9c", "9d", "9e", "9f",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af",
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
    "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf",
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7",
    "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf",
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "d8", "d9", "da", "db", "dc", "dd", "de", "df",
    "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7",
    "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef",
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff",
};

static char *
bintohex(unsigned int len, char *bindata)
{
    char *hexdata, *starthex;

    /* two chars per byte, plus null termination */
    starthex = hexdata = (char *)malloc(2*len + 1);
    if (!hexdata)
        return NULL;

    for (; len > 0; len--, bindata++) {
        register char *s = hex_table[(unsigned char)*bindata];
        *hexdata++ = s[0];
        *hexdata++ = s[1];
    }
    *hexdata = '\0';
    return starthex;
}
