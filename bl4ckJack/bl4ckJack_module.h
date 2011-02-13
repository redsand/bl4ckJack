
#ifndef __BL4CKJACK_MODULE__H_
#define __BL4CKJACK_MODULE__H_

#pragma once

#include <Qt>

#ifdef Q_WS_WIN
#define MY_EXPORT __declspec(dllexport)
#else
#define MY_EXPORT
#endif

#ifndef bl4ckJack_Module

//! Exported Bruteforcing Module Structure
/**
 * Exported Bruteforcing Module Structure
 * Manages Name, Credentials, Version, etc. of the provided dynamic modules
 */
typedef struct {

	//! Module Name
	char *name;

	//! Module Author(s)
	char *authors;

	//! Module Creation Date
	char *date;

	//! Version of Module
	float version;

	//! Module support salted passwords or not
	bool isSalted;

} bl4ckJack_Module;

#endif

//! Initialize Host
extern "C" MY_EXPORT void bl4ckJackInit(void);
typedef void (*fbl4ckJackInit)(void);

//! Match format support against input hash string
extern "C" MY_EXPORT bool bl4ckJackMatch(const char *);
typedef bool (*fbl4ckJackMatch)(const char *);

//! Get Module Info
extern "C" MY_EXPORT bl4ckJack_Module *bl4ckJackInfo(void);
typedef bl4ckJack_Module * (*fbl4ckJackInfo)(void);

//! Free initialized Host memory
extern "C" MY_EXPORT void bl4ckJackFree(void);
typedef void (*fbl4ckJackFree)(void);

//! Generate Hash
extern "C" MY_EXPORT void bl4ckJackGenerate(unsigned char *, size_t *, unsigned char *, size_t);
typedef void (*fbl4ckJackGenerate)(unsigned char *, size_t *, unsigned char *, size_t);

//! Initialize GPU for bruteforcing
extern "C" MY_EXPORT void bl4ckJackInitGPU(char *, int, void **, unsigned long , unsigned int);
typedef void (*fbl4ckJackInitGPU)(char *, int, void **, unsigned long , unsigned int);

//! GPU Bruteforcing Kernel Call (match happens inside)
extern "C" MY_EXPORT void bl4ckJackGenerateGPU(int, int, int, double *start, double *stop, char **successList, int *);
typedef void (*fbl4ckJackGenerateGPU)(int, int, int, double *start, double *stop, char **successList, int *);

//! Free initialized GPU memory
extern "C" MY_EXPORT void bl4ckJackFreeGPU(void);
typedef void (*fbl4ckJackFreeGPU)(void);

#endif