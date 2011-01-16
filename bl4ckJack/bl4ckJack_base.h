#ifndef __BL4CKJACK_BASECONVERSION_H__
#define __BL4CKJACK_BASECONVERSION_H__

#include <string>

class BaseConversion
 {

 public:
	 BaseConversion(std::string charset);

    //~BaseConversion();
	//! Return is _NOT_ thread-safe!
	char *ToBase(long double number);
	char *ToBase(long double number, char *buffer, size_t len);
	long double FromBase(char *);
	int getCharsetLength() {
		 return this->charsetLen;
	}

	const char* getCharset() {
		return this->charset.c_str();
	}

 private:
	 std::string charset;
	 int charsetLen;
	 char myBuf[1024];
	 void strrev(char *p);
	 

 };

#endif