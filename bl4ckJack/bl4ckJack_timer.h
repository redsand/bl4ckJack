#ifndef __BL4CKJACK_TIMER_H__
#define __BL4CKJACK_TIMER_H__

#pragma once

#include <Qt>

class Timer {
	public:
		Timer(void);
		~Timer(void);
		qint64 StartTiming(void);
		qint64 StopTiming(void);
		double ElapsedTiming(qint64 start, qint64 stop);
	private:
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
		qint64 m_Freq;   // Frequency for QueryPerformanceFrequency
#endif
};

#endif