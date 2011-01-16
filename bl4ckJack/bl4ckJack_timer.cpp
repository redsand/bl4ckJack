/* 
    Copyright (C) 2009  Benjamin Vernoux, titanmkd@gmail.com

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA */

/* Timer class for Win32 or Linux
*/
#include <time.h>
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	#include <windows.h> /* For Win32 QueryPerformanceCounter() */
#endif

#include "bl4ckJack_timer.h"


Timer::Timer(void)
{
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	QueryPerformanceFrequency((LARGE_INTEGER *)&m_Freq);
#endif
}

Timer::~Timer(void)
{
}

qint64 Timer::StartTiming(void)
{
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	qint64 i;
	if(this->m_Freq != 0)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)&i);
		return i;
	}else
		return clock();
#else
	return clock();
#endif
}

qint64 Timer::StopTiming(void)
{
	return Timer::StartTiming();
}

double Timer::ElapsedTiming(qint64 start, qint64 stop)
{ 
  // Returns elapsed time in ms
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	if(this->m_Freq != 0)
	{
		return ((double)(stop-start)/(double)m_Freq)*1000.0;
	}else
		return ((double)(stop-start)/(double)CLOCKS_PER_SEC)*1000.0; 
#else
	return ((double)(stop-start)/(double)CLOCKS_PER_SEC)*1000.0;  
#endif
}

