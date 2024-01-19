/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#ifndef __PROFILER_H__
#define __PROFILER_H__
#ifndef __XTENSA__
#undef PROFILE
#endif

#if defined(HW_SIM) && (1 == HW_SIM)
#undef PROFILE

#endif


#ifdef PROFILE
#include <stdio.h>
#include <string.h>
#include <sys/times.h>
#include <xtensa/sim.h>
#include <inttypes.h>

#define MAX_PROFILER_NAME_LENGTH 100
#define MAX_PROFILER_PARAMS_LENGTH 200
#define MAX_PROFILER_METRIC_UNITS_LENGTH 20

#define CCOUNT_AVAILABLE

#ifdef CCOUNT_AVAILABLE

static unsigned long inline GETCLOCK(void)
{
  unsigned long r;
  __asm__ volatile ("rsr.ccount %0" : "=r" (r));
  return r;
}

#define get_clock(val) *val = GETCLOCK() 
#define _START(prof) (&gProfiler[prof])->start
#define _STOP(prof)  (&gProfiler[prof])->stop
#define _EXCLUDE_START(prof) (&gProfiler[prof])->exclude_start
#define _EXCLUDE_STOP(prof)  (&gProfiler[prof])->exclude_stop

#else
#define get_clock(val) times(val) 
#define _START(prof) (&gProfiler[prof])->start.tms_utime
#define _STOP(prof)  (&gProfiler[prof])->stop.tms_utime
#define _EXCLUDE_START(prof) (&gProfiler[prof])->exclude_start.tms_utime
#define _EXCLUDE_STOP(prof)  (&gProfiler[prof])->exclude_stop.tms_utime
#endif
#define XTPWR_PROFILER_OPEN(prof, _name, _params, _metric_points, _metric_units, _metric_inverted) {              \
  (&gProfiler[prof])->cycles = (&gProfiler[prof])->exclude_cycles = 0;                                            \
  (&gProfiler[prof])->frame_cnt = (&gProfiler[prof])->peak_frame = 0;                                             \
  (&gProfiler[prof])->curr = (&gProfiler[prof])->sum  = (&gProfiler[prof])->ave = (&gProfiler[prof])->peak = 0.0; \
  (&gProfiler[prof])->curr_metric = (&gProfiler[prof])->ave_metric = 0.0;                                         \
  (&gProfiler[prof])->metric_points = _metric_points;                                                             \
  _START(prof) = 0;                                                                                               \
  _STOP(prof) = 0;                                                                                                \
  strcpy((&gProfiler[prof])->name , _name);                                                                       \
  strcpy((&gProfiler[prof])->params , _params);                                                                   \
  strcpy((&gProfiler[prof])->metric_units, (NULL != _metric_units) ? _metric_units : "");                         \
  (&gProfiler[prof])->metric_inverted = _metric_inverted;                                                         \
}

#define XTPWR_PROFILER_START( prof ) {        \
  xt_iss_profile_enable();                    \
  xt_iss_client_command("all", "enable");     \
  get_clock(&(&gProfiler[prof])->start);      \
}

#define XTPWR_PROFILER_STOP( prof ) {                         \
  get_clock(&(&gProfiler[prof])->stop);                       \
  xt_iss_client_command("all", "disable");                    \
  xt_iss_profile_disable();                                   \
  (&gProfiler[prof])->cycles += _STOP(prof) - _START(prof);   \
  _START(prof) = 0;                                           \
  _STOP(prof) = 0;                                            \
}

/* BASIC varisnts of the profiler, that do not force ISS mode */
#define XTPWR_BASIC_PROFILER_START( prof ) {        \
  xt_iss_profile_enable();                    \
  xt_iss_client_command("all", "enable");     \
  get_clock(&(&gProfiler[prof])->start);      \
}

#define XTPWR_BASIC_PROFILER_STOP( prof ) {                         \
  get_clock(&(&gProfiler[prof])->stop);                       \
  xt_iss_client_command("all", "disable");                    \
  xt_iss_profile_disable();                                   \
  (&gProfiler[prof])->cycles += _STOP(prof) - _START(prof);   \
  _START(prof) = 0;                                           \
  _STOP(prof) = 0;                                            \
}



#define XTPWR_PROFILER_PRINT( prof )                                                               \
  printf( " frame %d : %10.2f cycles; Total cycles : %10.2f cycles; %6.2f %s\n",                                         \
      (&gProfiler[prof])->frame_cnt-1, (&gProfiler[prof])->curr, (&gProfiler[prof])->sum, (&gProfiler[prof])->curr_metric, \
      (&gProfiler[prof])->metric_units); \

#define XTPWR_PROFILER_UPDATE( prof ) {                                                                             \
  (&gProfiler[prof])->cycles = (&gProfiler[prof])->cycles - (&gProfiler[prof])->exclude_cycles;                     \
  (&gProfiler[prof])->curr = ( (double)(&gProfiler[prof])->cycles ) ;                                               \
  (&gProfiler[prof])->sum += (&gProfiler[prof])->curr;                                                              \
  (&gProfiler[prof])->ave = (&gProfiler[prof])->sum / ((&gProfiler[prof])->frame_cnt+1);                            \
  if ((&gProfiler[prof])->peak < (&gProfiler[prof])->curr)                                                          \
  {                                                                                                                 \
    (&gProfiler[prof])->peak = (&gProfiler[prof])->curr;                                                            \
    (&gProfiler[prof])->peak_frame = (&gProfiler[prof])->frame_cnt;                                                 \
  }                                                                                                                 \
  if((&gProfiler[prof])->metric_inverted)                                                                           \
  {                                                                                                                 \
    (&gProfiler[prof])->curr_metric = ( (double)((&gProfiler[prof])->metric_points) / (&gProfiler[prof])->cycles ) ;\
    (&gProfiler[prof])->ave_metric = ( (&gProfiler[prof])->metric_points / (&gProfiler[prof])->ave );               \
  }                                                                                                                 \
  else                                                                                                              \
  {                                                                                                                 \
    (&gProfiler[prof])->curr_metric = ( (double)(&gProfiler[prof])->cycles / (&gProfiler[prof])->metric_points ) ;  \
    (&gProfiler[prof])->ave_metric = ((&gProfiler[prof])->ave / (&gProfiler[prof])->metric_points);                 \
  }                                                                                                                 \
  (&gProfiler[prof])->cycles = (&gProfiler[prof])->exclude_cycles = 0;                                              \
  (&gProfiler[prof])->frame_cnt += 1;                                                                               \
}

#define XTPWR_PROFILER_AVE_TOTAL( no_of_prof ) {                                                                    \
  int prof;                                                                                                         \
  double total_ave=0;                                                                                                 \
  for(prof=0;prof<no_of_prof;prof++)                                                                                \
  {                                                                                                                 \
    total_ave+= (&gProfiler[prof])->ave;                                                                            \
  }                                                                                                                 \
    printf("PROFILE_INFO_NN_MODEL,  total average cycles per frame=%-10.2f\n",                                                       \
        total_ave);                                                                                                 \
}

#define XTPWR_PROFILER_CLOSE( prof, pass_flag, verify_flag) {                                                                                            \
  if((&gProfiler[prof])->metric_units[0] != '\0')                                                                                           \
  {                                                                                                                                         \
    printf("PROFILE_INFO, %-25s, avg_cyc=%-10.2f, effective_metric=%-6.2f (%s), peak_cyc=%-10.2f, peak_frame=%-3d, result=%s, params: %s\n",\
        (&gProfiler[prof])->name, (&gProfiler[prof])->ave, (&gProfiler[prof])->ave_metric, (&gProfiler[prof])->metric_units,                \
        (&gProfiler[prof])->peak, (&gProfiler[prof])->peak_frame, pass_flag ? ((verify_flag) ? "pass" : "NA") : "fail", (&gProfiler[prof])->params);                    \
  }                                                                                                                                         \
  else                                                                                                                                      \
  {                                                                                                                                         \
    printf("PROFILE_INFO, %-25s, avg_cyc=%-10.2f, peak_cyc=%-10.2f, peak_frame=%-3d, result=%s, params: %s\n",                              \
        (&gProfiler[prof])->name, (&gProfiler[prof])->ave,                                                                                  \
        (&gProfiler[prof])->peak, (&gProfiler[prof])->peak_frame, pass_flag ? ((verify_flag) ? "pass" : "NA") : "fail", (&gProfiler[prof])->params);                    \
  }                                                                                                                                         \
}

// TODO
#define XTPWR_PROFILER_EXCLUDE_ON( prof )        \
  get_clock(&(&gProfiler[prof])->exclude_start); \
  XT_ISS_MEM_BANDWIDTH_PROFILER_STOP             \
  :
#define XTPWR_PROFILER_EXCLUDE_OFF( prof )  {                                       \
  get_clock(&(&gProfiler[prof])->exclude_stop);                                     \
  (&gProfiler[prof])->exclude_cycles += _EXCLUDE_STOP(prof) - _EXCLUDE_START(prof); \
  _EXCLUDE_START(prof) = 0;                                                         \
  _EXCLUDE_STOP(prof) = 0;                                                          \
  XT_ISS_MEM_BANDWIDTH_PROFILER_START                                               \
}

#define XT_ISS_SWITCH_MODE(mode) xt_iss_switch_mode(mode)

typedef struct _profiler_t
{
  char name[MAX_PROFILER_NAME_LENGTH];

#ifdef CCOUNT_AVAILABLE
  uint32_t cycles;
  uint32_t start;
  uint32_t stop;
  uint32_t exclude_cycles;
  uint32_t exclude_start;
  uint32_t exclude_stop;
#else
  clock_t cycles;
  struct tms start;
  struct tms stop;
  clock_t exclude_cycles;
  struct tms exclude_start;
  struct tms exclude_stop;
#endif
  int frame_cnt, peak_frame;
  double curr, sum, ave, peak;
  double curr_metric, ave_metric;
  unsigned int metric_points;

  char params[MAX_PROFILER_PARAMS_LENGTH];
  char metric_units[MAX_PROFILER_METRIC_UNITS_LENGTH];
  int metric_inverted;

} profiler_t;


#else /* PROFILE */

typedef struct _profiler_t
{
  char dummy;
} profiler_t;

#define XTPWR_PROFILER_OPEN(prof, _name, _params, _metric_points, _metric_units, _metric_inverted)
#define XTPWR_PROFILER_START( prof )
#define XTPWR_PROFILER_STOP( prof )
#define XTPWR_PROFILER_UPDATE( prof )
#define XTPWR_PROFILER_AVE_TOTAL( no_of_prof )
#define XTPWR_PROFILER_CLOSE( prof , pass_flag, verify_flag) 
#define XTPWR_PROFILER_PRINT( prof ) 

#define XTPWR_PROFILER_EXCLUDE_ON( prof )
#define XTPWR_PROFILER_EXCLUDE_OFF( prof )
#define XT_ISS_SWITCH_MODE(mode)

#endif /* PROFILE */
#ifdef PROF_ALLOCATE  //define in only one file
profiler_t gProfiler[7];
#else
extern profiler_t gProfiler[7];
#endif

#endif //__PROFILER_H__ 
