#ifndef MD5_GLOBAL_H
#define MD5_GLOBAL_H

#include <QtCore/qglobal.h>

#ifdef MD5_LIB
# define MD5_EXPORT Q_DECL_EXPORT
#else
# define MD5_EXPORT Q_DECL_IMPORT
#endif

/* POINTER defines a generic pointer type */
typedef unsigned char *POINTER;

#ifndef HAVE_UINT32_T
#  if SIZEOF_INT == 4
typedef unsigned int uint32_t;
#  elif SIZEOF_LONG == 4
typedef unsigned long int uint32_t;
#  endif
#endif

/* MD5 context. */
typedef struct {
  unsigned int state[4];                                   /* state (ABCD) */
  unsigned int count[2];        /* number of bits, modulo 2^64 (lsb first) */
  unsigned char buffer[64];                         /* input buffer */
} MD5_CTX;

void MD5Init (MD5_CTX *);
void MD5Update (MD5_CTX *, unsigned char *, unsigned int);
void MD5Final (unsigned char [16], MD5_CTX *);



#endif // MD5_GLOBAL_H
