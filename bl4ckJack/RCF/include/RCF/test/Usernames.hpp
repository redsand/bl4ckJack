
//******************************************************************************
// RCF - Remote Call Framework
//
// Copyright (c) 2005 - 2010, Delta V Software. All rights reserved.
// http://www.deltavsoft.com
//
// RCF is distributed under dual licenses - closed source or GPL.
// Consult your particular license for conditions of use.
//
// Version: 1.3
// Contact: jarl.lindrud <at> deltavsoft.com 
//
//******************************************************************************

#ifndef INCLUDE_RCF_TEST_USERNAMES_HPP
#define INCLUDE_RCF_TEST_USERNAMES_HPP

#include <fstream>
#include <string>

#include <RCF/util/AutoBuild.hpp>

class Usernames
{
public:
    std::string mLocalUsername;
    std::string mLocalPassword;
    std::string mLocalPasswordBad;
    std::string mAdUsername;
    std::string mAdPassword;
    std::string mAdPasswordBad;
    std::string mAdDomain;
};

bool getUsernames(Usernames & usernames)
{
    std::string whichFile = RCF_TEMP_DIR "sspi.txt";

    std::ifstream fin(whichFile.c_str());
    
    if (!fin)
    {
        return false;
    }
    
    fin >> usernames.mLocalUsername;
    fin >> usernames.mLocalPassword;
    fin >> usernames.mLocalPasswordBad;
    fin >> usernames.mAdUsername;
    fin >> usernames.mAdPassword;
    fin >> usernames.mAdPasswordBad;
    fin >> usernames.mAdDomain;
    
    assert(fin);
    
    if (!fin)
    {
        return false;
    }

    fin.close();

    return true;
}

#endif // ! INCLUDE_RCF_TEST_USERNAMES_HPP
