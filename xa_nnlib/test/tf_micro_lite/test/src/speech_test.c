/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xa_nnlib_standards.h"

#define MAX_CMD_LINE_LENGTH 1024
#define MAX_MEM_ALLOCS      100
#define XA_MAX_ARGS         30
#define XA_SCREEN_WIDTH     80

#ifndef PARAMFILE
#define PARAMFILE "paramfilesimple_tf_micro_lite.txt"
#endif

char pb_input_file_path[MAX_CMD_LINE_LENGTH];
char pb_output_file_path[MAX_CMD_LINE_LENGTH];
char pb_input_file_name[MAX_CMD_LINE_LENGTH];
char pb_output_file_name[MAX_CMD_LINE_LENGTH];

static void prefix_dirname(char *fname, char *dirname)
{
    if(strlen(dirname) > 0)
    {
        char tempname[MAX_CMD_LINE_LENGTH];
        strncpy(tempname, fname, MAX_CMD_LINE_LENGTH);
        // Remove additional / slash from directory name
        if(dirname[strlen(dirname)-1] == '/')
        {
            dirname[strlen(dirname)-1] = '\0';
        }
        snprintf(fname,MAX_CMD_LINE_LENGTH, "%s/%s", dirname, tempname);
    }
}
void prefix_inpdir_name(char *inp_file)
{
    prefix_dirname(inp_file, pb_input_file_path);
}

void prefix_outdir_name(char *out_file)
{
    prefix_dirname(out_file, pb_output_file_path);
}

#define SAMPLE_RATE 16000

#ifdef __XTENSA__
#include <xtensa/tie/xt_hifi2.h>
#ifdef __XCC__
#include <xtensa/hal.h>
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif
int frontendprocess_inference(void *in, void *out);
#if defined(__cplusplus)
}
#endif

#define SILENCE_INDEX 3
#define UNKNOWN_INDEX 4
#define YES_INDEX 5
#define NO_INDEX 6

int xa_tf_micro_lite_main_process(int argc, char **argv)
{
  short *input=NULL;
  short *output=NULL;
  FILE *fp, *fout;
 
  if(argc<3)
  {
    printf("Usage: %s <input-wav-file.wav> <test_result.txt>\n", argv[0]);
    return -1;
  }

  strncpy(pb_input_file_name, argv[1],MAX_CMD_LINE_LENGTH); 
  prefix_inpdir_name(pb_input_file_name);
  fp = fopen(pb_input_file_name, "rb");
  if (fp==NULL) {
      printf("Unable to open file '%s'\n", pb_input_file_name);
      return -1;
  }         

  strncpy(pb_output_file_name, argv[2],MAX_CMD_LINE_LENGTH); 
  prefix_outdir_name(pb_output_file_name);
  fout = fopen(pb_output_file_name, "w");
  if (fout==NULL) {
      printf("Unable to open file '%s'\n", pb_output_file_name);
      return -1;
  }         

  fseek(fp, 0, SEEK_SET);       
  input = (short *)malloc(SAMPLE_RATE*sizeof(short)); 
  output = (short *)malloc(SAMPLE_RATE*sizeof(short));
  if(input==NULL || output==NULL)
  { 
      printf("memory allocation failed\n");
      return -1;
  }
  fread(input, SAMPLE_RATE, sizeof(short), fp); 

  printf("frontend inference running\n");
  frontendprocess_inference(input, output);

  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;

  const int kYesIndex = 2;
  const int kNoIndex = 3;
  const int kUpIndex = 4;
  const int kDownIndex = 5;
  const int kLeftIndex = 6;
  const int kRightIndex = 7;
  const int kOnIndex = 8;
  const int kOffIndex = 9;
  const int kStopIndex = 10;
  const int kGoIndex = 11;
  
  fprintf(fout, "silence %d\n", output[kSilenceIndex]);
  fprintf(fout, "unknown %d\n", output[kUnknownIndex]);
  fprintf(fout, "yes %d\n", output[kYesIndex]);
  fprintf(fout, "no %d\n", output[kNoIndex]);
  fprintf(fout, "up %d\n", output[kUpIndex]);
  fprintf(fout, "down %d\n", output[kDownIndex]);
  fprintf(fout, "left %d\n", output[kLeftIndex]);
  fprintf(fout, "right %d\n", output[kRightIndex]);
  fprintf(fout, "on %d\n", output[kOnIndex]);
  fprintf(fout, "off %d\n", output[kOffIndex]);
  fprintf(fout, "stop %d\n", output[kStopIndex]);
  fprintf(fout, "go %d\n", output[kGoIndex]);

  free(input);
  free(output);
  fclose(fp);
  fclose(fout);
  printf("speech_test run successful \n\n");

  return 0;
}

// Set cache attribute to Write Back No Allocate when the last argument is -wbna

void set_wbna(int *argc, char *argv[])
{
    if ( *argc > 1 && !strcmp(argv[*argc-1], "-wbna") )
    {
#ifdef __XCC__
        extern char _memmap_cacheattr_wbna_trapnull;

        xthal_set_cacheattr((unsigned)&_memmap_cacheattr_wbna_trapnull);
#endif
        (*argc)--;
    }
}


/****************************************************************************
 *   Main function for test-wrapper                                         *
 ****************************************************************************/

int
main (int   argc, char *argv[])
{
    FILE *param_file_id = NULL;

	char curr_cmd[MAX_CMD_LINE_LENGTH] = {0};
    int fargc = 0, curpos = 0;
    int processcmd = 0;

	char fargv[XA_MAX_ARGS][MAX_CMD_LINE_LENGTH] = {{0}};
	char *pargv[XA_MAX_ARGS] = {NULL};

    memset(pb_input_file_path,  0, MAX_CMD_LINE_LENGTH);
    memset(pb_output_file_path, 0, MAX_CMD_LINE_LENGTH);

    // NOTE: set_wbna() should be called before any other dynamic
    // adjustment of the region attributes for cache.
    set_wbna(&argc, argv);

    //xa_testbench_error_handler_init();
    /* Library name version etc print */
    fprintf(stderr, "\n--------------------------------------------------------\n");
    fprintf(stderr, "TF Micro Lite Example Testbench\n");
    fprintf(stderr, "%s version %s\n",
            xa_nnlib_get_lib_name_string(),
            xa_nnlib_get_lib_version_string());
    fprintf(stderr, "Cadence Design Systems, Inc. http://www.cadence.com\n");
    fprintf(stderr, "--------------------------------------------------------\n");
    fprintf(stderr, "\n");


    if(argc == 1)
    {
        if ((param_file_id = fopen(PARAMFILE, "r")) == NULL )
        {
            printf("Parameter file \"%s\" not found.\n", PARAMFILE);
            printf("For Command line usage, Use -h \n");
            exit(0);
        }

        /* Process one line at a time */
        while(fgets(curr_cmd, MAX_CMD_LINE_LENGTH, param_file_id))
        {
            curpos = 0;
            fargc = 0;
            /* if it is not a param_file command and if */
            /* CLP processing is not enabled */
            if(curr_cmd[0] != '@' && !processcmd)     /* skip it */
            {
                continue;
            }
            // Reserver 0 for the binary name
            strncpy(fargv[0], argv[0], MAX_CMD_LINE_LENGTH);
            fargc++;

            while(sscanf(curr_cmd + curpos, "%s", fargv[fargc]) != EOF)
            {
                if(fargv[0][0]=='/' && fargv[0][1]=='/')
                    break;
                if(strcmp(fargv[0], "@echo") == 0)
                    break;
                if(strcmp(fargv[fargc], "@New_line") == 0)
                {
                    char * str = fgets(curr_cmd + curpos, MAX_CMD_LINE_LENGTH,
                                       param_file_id);
                    (void)str;
                    continue;
                }
                curpos += strlen(fargv[fargc]);
                while(*(curr_cmd + curpos)==' ' || *(curr_cmd + curpos)=='\t')
                    curpos++;
                fargc++;
            }

            if(fargc < 2) /* for blank lines etc. */
                continue;

            if(strcmp(fargv[1], "@Output_path") == 0)
            {
                if(fargc > 2) strcpy(pb_output_file_path, fargv[2]);
                else strcpy(pb_output_file_path, "");
                continue;
            }

            if(strcmp(fargv[1], "@Input_path") == 0)
            {
                if(fargc > 2) strcpy(pb_input_file_path, fargv[2]);
                else strcpy(pb_input_file_path, "");
                continue;
            }

            if(strcmp(fargv[1], "@Start") == 0)
            {
                processcmd = 1;
                continue;
            }

            if(strcmp(fargv[1], "@Stop") == 0)
            {
                processcmd = 0;
                continue;
            }

            /* otherwise if this a normal command and its enabled for execution */
            if(processcmd)
            {
                int i;
                for(i=0; i<fargc; i++)
                {
                    pargv[i] = fargv[i];
                }

                //MEM_init();

                xa_tf_micro_lite_main_process(fargc, pargv);

                //MEM_freeall();

            }
        }
    }
    else
    {
        //MEM_init();

        xa_tf_micro_lite_main_process(argc, &argv[0]); // TODO

        //MEM_freeall();
    }

    return 0; //XA_NO_ERROR;
} /* end main decode function */
