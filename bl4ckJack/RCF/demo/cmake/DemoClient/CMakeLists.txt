
ADD_DEFINITIONS( ${RCF_DEFINES} )

INCLUDE_DIRECTORIES( ${RCF_INCLUDES} )

SET(EXECUTABLE_OUTPUT_PATH ${BIN_PATH} )
SET(LIBRARY_OUTPUT_PATH ${BIN_PATH})
LINK_DIRECTORIES( ${BIN_PATH} )

ADD_EXECUTABLE(
    DemoClient
    ${RCF_ROOT}/demo/Client.cpp)

TARGET_LINK_LIBRARIES( DemoClient RcfLib )
