#ifndef SHA1_GLOBAL_H
#define SHA1_GLOBAL_H

#include <QtCore/qglobal.h>

#ifdef SHA1_LIB
# define SHA1_EXPORT Q_DECL_EXPORT
#else
# define SHA1_EXPORT Q_DECL_IMPORT
#endif

#include <sys/types.h>
#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

typedef struct {
  unsigned int state[5];
  unsigned int count[2];
  unsigned char buffer[64];
} SHA1_CTX;

void SHA1Transform(unsigned int state[5], const unsigned char buffer[64]);
void SHA1Init(SHA1_CTX* context);
void SHA1Update(SHA1_CTX* context, const unsigned char* data, unsigned int len);
void SHA1Final(unsigned char digest[20], SHA1_CTX* context);

# define SHA1_Transform SHA1Transform
# define SHA1_Init SHA1Init
# define SHA1_Update SHA1Update
# define SHA1_Final SHA1Final

# define SHA_DIGEST_LENGTH 20

#endif // SHA1_GLOBAL_H
