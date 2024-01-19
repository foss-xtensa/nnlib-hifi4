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
#ifndef __XA_PCM_RENDERER_CMDL_PARSE_H__
#define __XA_PCM_RENDERER_CMDL_PARSE_H__

// Macros for commandline parser
#define ARGTYPE_STRING(_switch, STRING, MAX_STRING_LENGTH)      \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
            if(argv[argidx+1]) {                           \
              /* printf("PARSE %s: %s %s\n", _switch, argv[argidx], argv[argidx+1]); */ \
              strncpy(STRING, argv[argidx+1], MAX_STRING_LENGTH);  \
              argidx++;\
            } \
            continue;\
          }

#define ARGTYPE_STRING_TO_ARRAY(_switch, STRING, MAX_STRING_LENGTH, ARRAY)      \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
            /* printf("PARSE %s: %s %s\n", _switch, argv[argidx], argv[argidx+1]); */ \
            strncpy(STRING, argv[argidx+1], MAX_STRING_LENGTH);  \
            char *token = strtok(STRING, " ");\
            int count = 0; \
            while ((token != NULL) && count < MAX_DIMS) \
            { \
              ARRAY[count] = atoi(token); \
              count++; \
              token = strtok(NULL, " "); \
            } \
            argidx++;\
            strncpy(STRING, argv[argidx], MAX_STRING_LENGTH);  \
            continue;\
          }

#define ARGTYPE_ONETIME_CONFIG_ARRAY(_switch, ARRAY, NUM_DIMS, STRING)      \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
            int count = 0; \
            while (count < NUM_DIMS) \
            { \
              ARRAY[count] = atoi(argv[argidx+1]); \
              if(STRING != NULL) \
              { \
                strcat(STRING, argv[argidx+1]); \
                strcat(STRING, " "); \
              } \
              argidx++; \
              count++; \
            } \
            continue;\
          }



#define ARGTYPE_INDICATE(_switch, _param)         \
   if(strcmp((argv[argidx]), _switch) == 0) {      \
         _param = 1;  \
         continue;\
       }
#define ARGTYPE_INDICATE_NEG(_switch, _param)         \
   if(strcmp((argv[argidx]), _switch) == 0) {      \
         _param = 0;  \
         continue;\
       }

#define ARGTYPE_ONETIME_CONFIG( _switch, _param)                \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
          _param = atoi(argv[argidx+1]);  \
      /* printf("PARSE: %s %d\n", argv[argidx], _param); */ \
          argidx++;\
          continue;\
        }

#define ARGTYPE_ONETIME_CONFIG_F32( _switch, _param)                \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
          _param = atof(argv[argidx+1]);  \
          argidx++;\
          continue;\
        }

#define ARGTYPE_CONFIG(_flag, _switch, _param)                \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
          _param = atoi(argv[argidx+1]);  \
          _flag |= 1;  \
          argidx++;\
          continue;\
        }

#define ARG_IGNORE_2(_switch)                \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
          argidx++;\
          continue;\
        }

#define ARG_IGNORE_1(_switch)                \
    if(strcmp((argv[argidx]), _switch) == 0) {             \
          continue;\
        }

#define ARGTYPE_WARN_AND_IGNORE2(_switch) \
      if(strcmp((argv[argidx]), _switch) == 0) {             \
                printf("%s is not supported in current mode of operation", _switch);\
                argidx++;\
                continue;\
              }

#define ARGTYPE_PARSE_STR(_flag, _switch, _callback_fn, _callback_handle) \
      if(strcmp((argv[argidx]), _switch) == 0) {             \
                _callback_fn(_callback_handle, argv[argidx+1]);\
                _flag |= 1;  \
                argidx++;\
                continue;\

#endif // __XA_PCM_RENDERER_CMDL_PARSE_H__

