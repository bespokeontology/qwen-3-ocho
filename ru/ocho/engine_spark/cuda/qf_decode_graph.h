// qf_decode_graph.h - CUDA Graph lifecycle for the fixed decode schedule
#pragma once
#include "../qwenflash.h"

int qf_graph_capture(QfModel *m);
int qf_graph_ready(void);
int qf_graph_step(QfModel *m, int token, long pos);
int qf_kv_reset(QfModel *m);
float qf_graph_last_ms(void);
void qf_graph_stats(long *steps, double *avg_ms);
void qf_graph_destroy(void);
