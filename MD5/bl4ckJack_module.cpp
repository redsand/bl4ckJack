
#include "stdafx.h"
#include <bl4ckJack_module.h>
#include "md5_global.h"
#include <Qt>
#include <qregexp.h>

static bl4ckJack_Module module[] = {
	"MD5",
	"Tim Shelton <redsand@blacksecurity.org>",
	"2010-12-15 00:00:00",
	1.0
};

//! Initialize
extern "C" MY_EXPORT void bl4ckJackInit(void) {

}

//! Match format support against input hash string
extern "C" MY_EXPORT bool bl4ckJackMatch(const char *match) {
	QRegExp rx("[0-9a-zA-Z]{32}$");
	int ret = rx.indexIn(match);
	if(ret < 0)
		return false;

	return true;
}

//! Get Module Info
extern "C" MY_EXPORT bl4ckJack_Module *bl4ckJackInfo(void) {
	return module;
}

//! Free initialized memory
extern "C" MY_EXPORT void bl4ckJackFree(void) {

}

//! Crunch
extern "C" MY_EXPORT void bl4ckJackGenerate(unsigned char *dst_buf, size_t *retLen, unsigned char *hash, size_t len) {

	MD5_CTX ctx;

	if(!dst_buf || !retLen || !hash) return;
	
	MD5Init(&ctx);
	MD5Update(&ctx, (unsigned char*)hash, len);
	MD5Final(dst_buf, &ctx);
	*retLen = 16;

}
