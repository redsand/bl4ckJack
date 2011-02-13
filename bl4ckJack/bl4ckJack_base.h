#ifndef __BL4CKJACK_BASECONVERSION_H__
#define __BL4CKJACK_BASECONVERSION_H__

#include <string>
//! BaseToX Conversion Class
/**
  * BaseToX Conversion Class responsible for converting our permutation # to our string
  */
class BaseConversion
 {

 public:
	
	  /**
	   * BaseConversion Constructor.
       * Instantiate our constructor and specify what character set will be used.
       * @param charset a character pointer.
       * @see class BaseConversion
       * @see ~BaseConversion()
       * @see ToBase()
       * @return None
       */
	   //! BaseConversion Constructor
	 BaseConversion(std::string charset);

	 //! BaseConversion Destructor
    ~BaseConversion() {
	}
	
	/**
	  * From Permutation Number to String.
	  * Convert permutating number into our string. Note, this uses a static buffer and is not thread safe.
	  *
      * @param long double
      * @see BaseConversion()
      * @see ~BaseConversion()
      * @see ToBase()
      * @return character pointer
      */
	  //! From Permutation Number to String. (NOT thread-safe)
	char *ToBase(long double number);
	
	/**
	  * From Permutation Number to String.
	  * Convert permutating number into our string.
	  * @param long double
	  * @param character pointer
	  * @param size_t length
      * @see BaseConversion()
      * @see ~BaseConversion()
      * @see ToBase()
      * @return character pointer
      */
	  //! From Permutation Number to String.
	char *ToBase(long double number, char *buffer, size_t len);
	
	/**
	  * From BaseX To Permutation Number.
	  * Convert string into permutation number.
	  * @param character pointer
      * @see BaseConversion()
      * @see ~BaseConversion()
      * @see ToBase()
      * @return long double
	  */
	  //! From BaseX To Permutation Number.
	long double FromBase(char *);
	
	/**
	  * Get Character Set Length.
	  * Convert string into permutation number.
      * @see BaseConversion()
      * @see charset;
      * @return int
	  */
	  //! Get Character Set Length.
	int getCharsetLength() {
		 return this->charsetLen;
	}

	/**
	  * Get Character Set
	  * Return character set specified.
	  * @param character pointer
      * @see BaseConversion()
      * @see ~BaseConversion()
      * @return constant character pointer
	  */
	  //! Get Character Set
	const char* getCharset() {
		return this->charset.c_str();
	}

 private:
	 std::string charset;
	 int charsetLen;
	 char myBuf[1024];
	 
	//! String Reverse
	/**
	  * String Reverse
	  * Reverse the letters in an existing buffer provided.
	  * @see class BaseConversion
      * @see ~BaseConversion()
      * @return None
	  */
	 void strrev(char *p);

 };

#endif