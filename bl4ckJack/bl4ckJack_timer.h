#ifndef __BL4CKJACK_TIMER_H__
#define __BL4CKJACK_TIMER_H__

#pragma once

#include <Qt>

//! Timer Class
/**
 * Timer Class
 * Timer Class used for detecting and managing time elapse.
 */
class Timer {
	public:
		//! Timer Constructor
		/**
		  * Timer Constructor
		  * Used for detecting and managing time elapse.
		  * @see Timer()
		  * @see ~Timer()
		  * @return None
		  */
		Timer(void);
		
		//! Timer Deconstructor
		/**
		  * Timer Deconstructor
		  * Used for detecting and managing time elapse.
		  * @see Timer()
		  * @see ~Timer()
		  * @return None
		  */
		~Timer(void);
		
		//! Timer Start Timing
		/**
		  * Timer Start Timing
		  * Used to begin the timeing process
		  * @see Timer()
		  * @see StopTiming()
		  * @see ElapsedTiming()
		  * @see ~Timer()
		  * @return qint64
		  */
		qint64 StartTiming(void);
		
		//! Timer Stop Timing
		/**
		  * Timer Stop Timing
		  * Used to end the timeing process
		  * @see Timer()
		  * @see StartTiming()
		  * @see ElapsedTiming()
		  * @see ~Timer()
		  * @return qint64
		  */
		qint64 StopTiming(void);
		
		//! Timer Calculate Elapsed Time
		/**
		  * Timer Calculate Elapsed Time
		  * Used to calculate the elapsed time between start and stop
		  * @param qint64 start
		  * @param qint64 stop
		  * @see Timer()
		  * @see StartTiming()
		  * @see StopTiming()
		  * @see ~Timer()
		  * @return double
		  */
		double ElapsedTiming(qint64 start, qint64 stop);
	private:
	
#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
		qint64 m_Freq;   // Frequency for QueryPerformanceFrequency
#endif
};

#endif