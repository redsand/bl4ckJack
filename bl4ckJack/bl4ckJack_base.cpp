
#include "bl4ckJack_base.h"
#include <Qt>
#include <QDebug>

#include <stdio.h>
#include <math.h>

BaseConversion::BaseConversion(std::string charset) {
	qDebug() << "Using local charset " << charset.c_str();
	this->charset = charset;
	this->charsetLen = charset.length();
}

//~BaseConversion();

// Return is _NOT_ thread-safe!
char * BaseConversion::ToBase(long double number) {
	return ToBase(number, this->myBuf, 1023);
}

char * BaseConversion::ToBase(long double number, char *buffer, size_t len) {
		int r = 0;
		int iter=0;
		number -= 1;
		
		if(number < 0) {
			buffer[0] = '\0';
			return buffer;
		}

		do {

			if(iter > (len-1)) break;
			
			r = floor(fmod(number, (long double)this->charsetLen)); //number % (this->charsetLen) /*remainder*/ /*drem*/ /*fmod(number, this->charsetLen); */;
			
			if(r < this->charsetLen)
				buffer[iter++] = this->charset[r];
			else
				buffer[iter++] = '=';

			number = floor((number / (this->charsetLen))) - 1;
			//qDebug() << "number is " << (double) number;
		} while(number >= 0);

		buffer[iter] = '\0';
		this->strrev(buffer);
		//qDebug() << "Base string is " << buffer;
		return buffer;
}

long double BaseConversion::FromBase(char *input) {

	int r = strlen(input), s=0;
	int iter=0, i=0;
	long double ret=0;		
		
	if(!input) return 0;
	
	while(input[iter]) {
		for(i=0; i < this->charsetLen; i++)
			if(this->charset[i] == input[iter])
				break;

		if(i >= this->charsetLen) {
			// this failed;
			break;
		}

		s = (this->charset[0] - this->charset[i]);
		qDebug() << "Between " << this->charset[0] << " and " << this->charset[i] << " is " << s;
		ret += (s / (long double) pow((long double) this->charsetLen, (int)iter + 1) );
		iter++;
	}

	qDebug() << "finalized with " << (double) ret;
	return ret;
}

void BaseConversion::strrev(char *p) {
  char *q = p;
  while(q && *q) ++q;
  for(--q; p < q; ++p, --q)
	*p = *p ^ *q,
	*q = *p ^ *q,
	*p = *p ^ *q;
}
