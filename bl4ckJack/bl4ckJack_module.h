
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

typedef struct {

	char *name;
	char *authors;
	char *date;
	float version;

} bl4ckJack_Module;

#endif

//! Initialize
extern "C" MY_EXPORT void bl4ckJackInit(void);
typedef void (*fbl4ckJackInit)(void);

//! Match format support against input hash string
extern "C" MY_EXPORT bool bl4ckJackMatch(const char *);
typedef bool (*fbl4ckJackMatch)(const char *);

//! Get Module Info
extern "C" MY_EXPORT bl4ckJack_Module *bl4ckJackInfo(void);
typedef bl4ckJack_Module * (*fbl4ckJackInfo)(void);


//! Free initialized memory
extern "C" MY_EXPORT void bl4ckJackFree(void);
typedef void (*fbl4ckJackFree)(void);

//! Generate
extern "C" MY_EXPORT void bl4ckJackGenerate(unsigned char *, size_t *, unsigned char *, size_t);
typedef void (*fbl4ckJackGenerate)(unsigned char *, size_t *, unsigned char *, size_t);


#endif