/* required MPI include file */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define MASTER 0

int main (int argc, char *argv[])
{
int numtasks, taskid, len;
char hostname[MPI_MAX_PROCESSOR_NAME];

// initialize MPI
MPI_Init(&argc, &argv);

// get number of tasks
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

// get my rank
MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

MPI_Get_processor_name(hostname, &len);

printf("Hello from task %d on %s!\n", taskid, hostname);

if (taskid == MASTER)
	printf("MASTER: Number of MPI tasks is: %d\n", numtasks);

MPI_Finalize();

}