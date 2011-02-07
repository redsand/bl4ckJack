
#include "bl4ckJack_bruteforce.h"
#include "bl4ckJack_distributed.h"

#include "bl4ckJack_btree.h"
#include "bl4ckJack.h"
#include "bl4ckJack_timer.h"

#include "cuda_gpu.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <Qt>
#include <QString>
#include <QDebug>

static char *
bintohex(unsigned int len, char *bindata);
__inline BOOL hyperThreadingOn();

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
	

	
	std::string charset = settings->value("config/current_charset","empty charset").toString().toStdString();
	this->base = new BaseConversion(charset);

	// lets check out our cuda availability
	GPU_Dev gpu;
	int gpu_count = gpu.getDevCount();

	// need to create a thread per cpu and pass arg to thread as offset

	this->hyperThreading = hyperThreadingOn();

	// per CPU single cpu then:
	// http://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
#ifdef WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );
	this->cpuCount = sysinfo.dwNumberOfProcessors;

	
   if(!SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
   {
      DWORD dwError = GetLastError();
   } 

#else
	this->cpuCount = sysconf( _SC_NPROCESSORS_ONLN );
#endif

	// for some reason this is 2x as fast when we run under 1 thread:
	this->cpuCount = 1;

	int cpu_stop = 0;
	//this->cpuCount *= 4;
	if(this->hyperThreading) {
		if( (this->cpuCount/2) - 1 <= 0) {
			cpu_stop = this->cpuCount;
		} else {
			cpu_stop = (this->cpuCount/2) - 1;
			this->cpuCount = cpu_stop;
		}
		for(int cpu_iter=0; cpu_iter < cpu_stop; cpu_iter++) {
			qDebug() << "Initiating " << cpu_iter << " CPU threads";
			cpuCurrent.push_back(cpu_iter);
#ifdef WIN32
		hThreadCPU = CreateThread(NULL, 0, BruteForce::NewThreadCPU, this, 0, &threadIdCPU);
#else
		pthread_create(&threadIdCPU, BruteForce::NewThreadCPU, this);
#endif

		}
	} else {	
		for(int cpu_iter=0; cpu_iter < (this->cpuCount)+1; cpu_iter++) {
			qDebug() << "Initiating " << cpu_iter << " CPU threads";
			cpuCurrent.push_back(cpu_iter);
#ifdef WIN32
			hThreadCPU = CreateThread(NULL, 0, BruteForce::NewThreadCPU, this, 0, &threadIdCPU);
#else
			pthread_create(&threadIdCPU, BruteForce::NewThreadCPU, this);
#endif

		}
	}

#if defined(WIN32) || defined(WIN64)
	/*
	HANDLE hProcessThread = OpenProcess(PROCESS_ALL_ACCESS, 0, GetCurrentProcessId());
	SetProcessAffinityMask(hProcessThread, this->cpuCount);
	CloseHandle(hProcessThread);
	*/
	
#else
#endif

#ifdef WIN32
	hThreadGPU = CreateThread(NULL, 0, BruteForce::NewThreadGPU, this, 0, &threadIdGPU);
#else
	pthread_create(&threadIdGPU, BruteForce::NewThreadGPU, this);
#endif

	while(!stopRunning) {

		int keyspaceIter = 0;
		std::list<std::pair<long double, long double> >::iterator iter;
		//for(keyspaceIter = 0; keyspaceIter < keyspaceList->size(); keyspaceIter++) {
		//for(iter = keyspaceList.begin(); iter != keyspaceList.end(); iter++) {
		iter = keyspaceList.begin();
		long double current_iter = 0;
		int pct=0;
		while(iter != keyspaceList.end()) {
			std::pair<long double, long double> pair = *iter;

			if(current_iter > 0)
				pair.first = current_iter; // in case our previous token overlapped

			// if no gpu then set this to 100
			if(gpu_count <= 0) {
				pct = 100;
			} else {
				pct = settings->value("config/dc_cpu_keyspace_pct", 10).toInt();
				if(pct <= 0) pct = 5;
			}

			long double space = pair.second - pair.first;
			long double cpu_amnt = (long double) (space * (float)((float)pct / 100));
			long double gpu_amnt = 100 - cpu_amnt;

			// check and make sure we have gpus to use, otherwise we're going to rely strictly on CPU

			long double cur = pair.second - pair.first;

			long double final_second = 0;

			// re-check and verify our list of keyspaces is being calculated correctly
			// keep getting 0 through 1st set of second
			
			int breakIter=0;
			while(cur > 0) {
				long double sub_amnt = (long double) floor((cpu_amnt / cpuCount));
				std::pair< long double, long double> pair2;
				for(int mine = 0; mine < cpuCount; mine++) {
					pair2.first = ((long double)pair.first + ((mine + breakIter) * sub_amnt));
					pair2.second = ((long double) pair.first + ((mine + breakIter) * sub_amnt) + sub_amnt + 1);
 					CPUkeyspaceList.push_back(pair2);
					cur -= (pair2.second - pair2.first);
				}
				final_second = pair2.second;

				// per GPU
				// if gpu available {
				/*
				sub_amnt = (long double) floor((gpu_amnt / gpuCount));
				for(int mine = 0; mine < cpuCount; mine++) {
					pair.first = ((long double)keyspaceList->at(keyspaceIter).first + ((mine + breakIter) * sub_amnt));
					pair.second = ((long double) keyspaceList->at(keyspaceIter).first + ((mine + breakIter) * sub_amnt) + sub_amnt);
 					GPUkeyspaceList.push_back(pair);
					cur -= (pair.second - pair.first);
					final_second = pair.second;
				}
				*/

				breakIter++;
			}

			current_iter = final_second + 1;

			//qDebug() << "First " << (double)keyspaceList->at(keyspaceIter).first << " < " << (double)keyspaceList->at(keyspaceIter).second;
			if(current_iter >= pair.second) {
				iter++;
				keyspaceIter++;
			}

			qDebug() << "CPUKeyspaceList size " << CPUkeyspaceList.size();
		}
		keyspaceList.clear();

#ifdef WIN32
		Sleep(1500);
#else
		usleep(1500);
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
unsigned long _stdcall BruteForce::NewThreadCPU(void *param) {
#else
void *BruteForce::NewThreadCPU(void *param) {
#endif
	BruteForce *self = (BruteForce*) param;
	
#if defined(WIN32) || defined(WIN64)
	return self->NewThreadCPU(self, GetCurrentThreadId());
#else
	return self->NewThreadCPU(self, pthread_self());
#endif
}


// THIS FUNCTION SHOULD BE LAUNCHED PER CPU!
#ifdef WIN32
unsigned long BruteForce::NewThreadCPU(void *param, int thread_id) {
#else
void *BruteForce::NewThreadCPU(void *param, int thread_id) {
#endif

	Timer t;
	int hashFound = 0;

	// get our thread/cpu #
	BruteForce *self = (BruteForce*) param;
	self->cpuThreads.push_back(thread_id);

	// sleep 1 second so all threads catch up
#if defined(WIN32) || defined(WIN64)
	Sleep(1000);
#else
	usleep(1000);
#endif

	int my_cpu = -1;
	for(int cpu_iter=0; cpu_iter < this->cpuCount; cpu_iter++) {
		if(this->cpuThreads[cpu_iter] == thread_id) {
			my_cpu = cpu_iter;
			break;
		}
	}

	if(my_cpu < 0) {
		qDebug() << "cpu assignment failed, wtf?";
		my_cpu = 0;
	}

	unsigned char bruteStr[MAX_BRUTE_CHARS+1];
	unsigned char results[MAX_BRUTE_CHARS+1];

	memset(bruteStr, 0, MAX_BRUTE_CHARS + 1);
	memset(results, 0, MAX_BRUTE_CHARS + 1);

	// this thread is responsible for bruteforcing CPU

	// identify our keyspace and iterate through our permutations (non recursively)

	this->stats.milHashSec = 0; // hash/sec in millions
	this->stats.totalHashFound = 0;
	this->stats.currentOpenTokens = 0;

	while (!stopRunning) {
	
		long double currentStartToken=0, currentStopToken=0;
		std::list<std::pair<long double, long double>>::iterator iter;
		std::pair<long double, long double> token;

		//for(CPUkeyspaceIter = 0; CPUkeyspaceIter < CPUkeyspaceList->size(), !stopRunning; CPUkeyspaceIter++) {
		while(!stopRunning && !CPUkeyspaceList.empty()) {
			// load our charset and keyspace and begin bruteing

			currentStartToken=0;
			currentStopToken=0;

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
				//qDebug() << "START CPUKeyspaceList size " << CPUkeyspaceList.size();
				if(CPUkeyspaceList.size() > my_cpu) {	
					iter = CPUkeyspaceList.begin();
					for(int my_temp=0; my_temp < my_cpu; my_temp++)
						iter++;
					// ^^^^ instead of below cuz it wont let me
					//iter += my_cpu;
					token = *iter;
				} else {
					// we're finish
					break;
				}
				currentStartToken = token.first;
				size_t retLen=0;
				BOOL setKey = TRUE;
	
				while(!stopRunning && token.first <= token.second) {
					this->base->ToBase(token.first, (char *)bruteStr, MAX_BRUTE_CHARS);
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
						// if this is our last one, lets stop
					}
					
					//if(fmod(token->first, 60000) == 0) {
					{
						//qDebug() << "checking time elapse";
						if(t.ElapsedTiming(startTime, t.StopTiming()) >= 1000) {
							
							// warn that we're in need of more keys
							if(setKey) {
								if(CPUkeyspaceList.size() == 1) {
									if( (((token.second - token.first) * 0.80 ) ) < token.first) {
   										this->getKeyspace = true;
										setKey = FALSE;
									}
								}
							} else {	
								if(CPUkeyspaceList.size() == 1) {
									setKey = TRUE;
								}
							}

							//qDebug() << "time elapse success " << startTime << " and " << t.StopTiming() << " is " << t.ElapsedTiming(startTime, t.StopTiming());
							this->statsMutex.lock();
							cpuCurrent[my_cpu] = token.first;
							this->stats.totalHashFound += hashFound;
							this->stats.milHashSec += (((token.first - currentStartToken)) / 1000000);
							this->statsMutex.unlock();
							hashFound = 0;
							currentStopToken = token.first;
							currentStartToken = token.first;
							startTime = t.StartTiming();
						}
					} 
					
					token.first++;
				}
			}
			CPUkeyspaceList.remove(token); //pop_front();
			qDebug() << "NEW CPUKeyspaceList size " << CPUkeyspaceList.size();
		}

		this->getKeyspace = true;
		if(stopRunning) break;
#ifdef WIN32
		Sleep(1500);
#else
		usleep(1500);
#endif

	}

	stopRunning = FALSE;

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
#if defined(WIN32) || defined(WIN64)
	return self->NewThreadGPU(self, GetCurrentThreadId());
#else
	return self->NewThreadGPU(self, pthread_self());
#endif
}

#ifdef WIN32
unsigned long BruteForce::NewThreadGPU(void *param, int thread_id) {
#else
void *BruteForce::NewThreadGPUGPU(void *param, int thread_id) {
#endif

	
	Timer t;
	int hashFound = 0;

	// get our thread/cpu #
	BruteForce *self = (BruteForce*) param;

	while (!stopRunning) {
	
		long double currentStartToken=0, currentStopToken=0;
		std::list<std::pair<long double, long double>>::iterator iter;
		std::pair<long double, long double> token;

		while(!stopRunning && !GPUkeyspaceList.empty()) {
			// load our charset and keyspace and begin bruteing

			currentStartToken=0;
			currentStopToken=0;

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
				iter = GPUkeyspaceList.begin();
				token = *iter;
				currentStartToken = token.first;
				size_t retLen=0;
				BOOL setKey = TRUE;
	
				while(!stopRunning && token.first <= token.second) {


					// we have our starting and stopping token,
					// lets batch up our gpu kernel and store results
					//setup our arguments
					// cudaConfigureCall(blocks,threads,shared_mem_size,stream)
					// cudaSetupArgument(sth1,offset);
					// offset+=sizeof(sth1);
					// cudaSetupArgument(sth2,offset);

					//cudaLaunch("bl4ckJackGPUKernelExecute");

					dim3 grid, block;
					cudaError_t err;

					grid.x = blocks_x; grid.y = blocks_y; grid.z = 1;
					block.x = threads_per_block; block.y = 1; block.z = 1;
					//GPUBruteforceKernelExecute<<<grid, block, shared_mem_required>>>(src_gwords, dst_gwords);	

					err = cudaThreadSynchronize();

					/*
					this->base->ToBase(token.first, (char *)bruteStr, MAX_BRUTE_CHARS);
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
						// if this is our last one, lets stop
					}
					*/
				}
			}
			GPUkeyspaceList.remove(token); //pop_front();
		}

		this->getKeyspace = true;
		if(stopRunning) break;
#ifdef WIN32
		Sleep(1500);
#else
		usleep(1500);
#endif

	}

	stopRunning = FALSE;

#ifdef WIN32
	ExitThread(0);
	return 0;
#else
	pthread_exit(0);
	return 0;
#endif

}

void BruteForce::start() {

	
	// qDebug()

#ifdef WIN32
	hThread = CreateThread(NULL, 0, BruteForce::NewThread, this, 0, &threadId);
#else
	pthread_create(&threadId, BruteForce::NewThread, this);
#endif

}

void BruteForce::stop() {

	qDebug() << "received stop message";
	stopRunning = TRUE;
	Sleep(500);

#ifdef WIN32
	TerminateThread(this->hThread, 0);
	TerminateThread(this->hThreadCPU, 0);
	TerminateThread(this->hThreadGPU, 0);
#else
	pthread_cancel(this->threadId);
	pthread_cancel(this->threadIdCPU);
	pthread_cancel(this->threadIdGPU);
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

__inline BOOL hyperThreadingOn()
{
    DWORD       rEbx, rEdx;
    __asm {
         push   eax         // save registers used
         push   ebx
         push   ecx
         push   edx
         xor    eax,eax     // cpuid(1)
         add    al, 0x01
        _emit   0x0F
        _emit   0xA2
         mov    rEdx, edx   // Features Flags, bit 28 indicates if HTT (Hyper-Thread Technology) is
                            //  available, but not if it is on; if on, Count of logical processors > 1.
         mov    rEbx, ebx   // Bits 23-16: Count of logical processors.
                            //             Valid only if Hyper-Threading Technology flag is set.
         pop    edx         // restore registers used
         pop    ecx
         pop    ebx
         pop    eax
    }
    return (rEdx & (1<<28)) && (((rEbx & 0x00FF0000) >> 16) > 1);
} 
