/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

double *alloc1D(int m,int n);
void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
 
 int nprocs = 1, myrank = 0, i, j;
#ifdef _MPI_
 MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
 MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

 //if(!cb.noComm) {
	//MPI_Ibcast(E, (m+2)*(n+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Request reqE_prev, reqR;
	//MPI_Ibcast(E_prev, (m+2)*(n+2), MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqE_prev);
	//MPI_Ibcast(R, (m+2)*(n+2), MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqR);
	
	//if(myrank!=0)
	//{
		//MPI_Wait(&reqE_prev, MPI_STATUS_IGNORE);
		//MPI_Wait(&reqR, MPI_STATUS_IGNORE);
	//}
 //}
 
 double *E_local, *R_local, *E_prev_local, *E_prev_packed, *R_packed, *E_prev_local_packed, *R_local_packed;
 int procDataCnt[nprocs], displacement[nprocs];
 int m_local = cb.m/cb.py, n_local = cb.n/cb.px;
 int row = myrank/cb.px, col = myrank%cb.px;
 
 if(cb.m%cb.py != 0 && row < cb.m%cb.py)
	 m_local++;
 if(cb.n%cb.px != 0 && col < cb.n%cb.px)
	 n_local++;
 
 E_local = alloc1D(m_local+2,n_local+2);
 E_prev_local = alloc1D(m_local+2,n_local+2);
 R_local = alloc1D(m_local+2,n_local+2);
 E_prev_packed = alloc1D(cb.m,cb.n);
 R_packed = alloc1D(cb.m,cb.n);
 E_prev_local_packed = alloc1D(m_local, n_local);
 R_local_packed = alloc1D(m_local, n_local);
 
 int rowOffset = row*m_local, colOffset = col*n_local;
 
 if(row >= cb.m%cb.py)
	 rowOffset += cb.m%cb.py;
 
 if(col >= cb.n%cb.px)
	 colOffset += cb.n%cb.px;
  
 if(myrank==0)
 {	
	int loc=0;
	
	for(int p=0; p<nprocs; p++)
	{
		int m_localRecv = cb.m/cb.py, n_localRecv = cb.n/cb.px;
		int rowRecv = p/cb.px, colRecv = p%cb.px;
	 
		if(cb.m%cb.py != 0 && rowRecv < cb.m%cb.py)
			m_localRecv++;
		if(cb.n%cb.px != 0 && colRecv < cb.n%cb.px)
			n_localRecv++;
		
		int rowOffsetRecv = rowRecv*m_localRecv, colOffsetRecv = colRecv*n_localRecv;
 
		if(rowRecv >= cb.m%cb.py)
			rowOffsetRecv += cb.m%cb.py;
		 
		if(colRecv >= cb.n%cb.px)
			colOffsetRecv += cb.n%cb.px;
		
		procDataCnt[p] = m_localRecv*n_localRecv;
		displacement[p] = loc;
		
		for(int j=1; j<m_localRecv+1; j++)
		{
			for(int i=1; i<n_localRecv+1; i++)
			{
				E_prev_packed[loc] = E_prev[(rowOffsetRecv+j)*(cb.n+2) + colOffsetRecv+i];
				R_packed[loc] = R[(rowOffsetRecv+j)*(cb.n+2) + colOffsetRecv+i];
				loc++;
			}
		}
	}
 }
 
 if(!cb.noComm)
 {
	MPI_Scatterv(E_prev_packed, procDataCnt, displacement, MPI_DOUBLE, E_prev_local_packed, m_local*n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(R_packed, procDataCnt, displacement, MPI_DOUBLE, R_local_packed, m_local*n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	for(int j=1; j<m_local+1; j++)
	{
		for(int i=1; i<n_local+1; i++)
		{
			//E_local[j*(n_local+2)+i] = E[(rowOffset+j)*(cb.n+2) + colOffset+i];
			E_prev_local[j*(n_local+2)+i] = E_prev_local_packed[(j-1)*n_local + i-1];
			R_local[j*(n_local+2)+i] = R_local_packed[(j-1)*n_local + i-1];
		}
	}
 }
 else
 {
	for(int j=1; j<m_local+1; j++)
	{
		for(int i=1; i<n_local+1; i++)
		{
			//E_local[j*(n_local+2)+i] = E[(rowOffset+j)*(cb.n+2) + colOffset+i];
			E_prev_local[j*(n_local+2)+i] = E_prev[(rowOffset+j)*(cb.n+2) + colOffset+i];
			R_local[j*(n_local+2)+i] = R[(rowOffset+j)*(cb.n+2) + colOffset+i];
		}
	}
 }

 MPI_Request req[8]; //, req2, req3, req4, req5, req6, req7, req8;
 MPI_Datatype column_type; 
 MPI_Type_vector(m_local, 1, n_local+2, MPI_DOUBLE, &column_type);
 MPI_Type_commit(&column_type);

 #endif

 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
#ifdef _MPI_
	switch (row) {
		// Fills in the TOP Ghost Cells
		case 0: {
			for (i = 0; i < (n_local+2); i++) {
				E_prev_local[i] = E_prev_local[i + (n_local+2)*2];
			}
			break;
		}
	}
	// Fills in the BOTTOM Ghost Cells
	if (row == cb.py-1) {
		for (i = ((m_local+2)*(n_local+2)-(n_local+2)); i < (m_local+2)*(n_local+2); i++) {
			E_prev_local[i] = E_prev_local[i - (n_local+2)*2];
		}
	}

	switch (col) {
		// Fills in the LEFT Ghost Cells
		case 0: {
			for (i = 0; i < (m_local+2)*(n_local+2); i+=(n_local+2)) {
				E_prev_local[i] = E_prev_local[i+2];
			}
			break;
		}
	}

	// Fills in the RIGHT Ghost Cells
	if (col == cb.px-1) {
		for (i = (n_local+1); i < (m_local+2)*(n_local+2); i+=(n_local+2)) {
			E_prev_local[i] = E_prev_local[i-2];
		}
	}
	
	if(!cb.noComm)
	{
		int at = 0;
		
		if (cb.px > 1) {
			if (col == 0) {
				MPI_Isend(E_prev_local+2*n_local+2, 1, column_type, myrank+1, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+2*n_local+3, 1, column_type, myrank+1, 0, MPI_COMM_WORLD, req+(at++));
			}

			else if (col == cb.px-1) {
				MPI_Irecv(E_prev_local+n_local+2, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Isend(E_prev_local+n_local+3, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, req+(at++));
			}

			else {
				MPI_Isend(E_prev_local+2*n_local+2, 1, column_type, myrank+1, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+n_local+2, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Isend(E_prev_local+n_local+3, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+2*n_local+3, 1, column_type, myrank+1, 0, MPI_COMM_WORLD, req+(at++));
			}
		}

		if (cb.py > 1) {
			if (row == 0) {
				MPI_Isend(E_prev_local+m_local*(n_local+2)+1, n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+(m_local+1)*(n_local+2)+1, n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, req+(at++));
			}

			else if (row == cb.py-1) {
				MPI_Irecv(E_prev_local+1, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Isend(E_prev_local+n_local+3, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, req+(at++));
			}

			else {
				MPI_Isend(E_prev_local+m_local*(n_local+2)+1, n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+1, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Isend(E_prev_local+n_local+3, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, req+(at++));
				MPI_Irecv(E_prev_local+(m_local+1)*(n_local+2)+1, n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, req+(at++));
			}
		}

		for (i=0;i<at;i++) MPI_Wait(req+i, MPI_STATUS_IGNORE);

		// // left to right data movement
		// if(col != cb.px-1)
		// {
		// 	// Send right most inner column to the process to the right		
		// 	MPI_Isend(&E_prev_local[2*n_local+2], 1, column_type, myrank+1, 0, MPI_COMM_WORLD, &req[0]);
		// }
		
		// if(col != 0)
		// {
		// 	// Receive in left ghost column from the process to the left
		// 	MPI_Irecv(E_prev_local+n_local+2, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, &req[1]);
		// }
		
		// // top to bottom data movement
		// if(row != cb.py-1)
		// {
		// 	// Send bottom most row to the process below
		// 	MPI_Isend(&E_prev_local[m_local*(n_local+2)+1], n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, &req[2]);
		// }
		
		// if(row != 0)
		// {
		// 	// Receive in top ghost row from the process above
		// 	MPI_Irecv(E_prev_local+1, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, &req[3]);
		// }
		
		// // right to left data movement	
		// if(col != 0)
		// {
		// 	// Send left most column to the process to the left
		// 	MPI_Isend(E_prev_local+n_local+3, 1, column_type, myrank-1, 0, MPI_COMM_WORLD, &req[4]);
		// }
		
		// if(col != cb.px-1)
		// {
		// 	// Receive in right ghost column from the process to the right
		// 	MPI_Irecv(&E_prev_local[2*n_local+3], 1, column_type, myrank+1, 0, MPI_COMM_WORLD, &req[5]);
		// }
		
		// // bottom to top data movement
		// if(row != 0)
		// {
		// 	// Send top most row to the process above
		// 	MPI_Isend(E_prev_local+n_local+3, n_local, MPI_DOUBLE, myrank-cb.px, 0, MPI_COMM_WORLD, &req[6]);
		// }
		
		// if(row != cb.py-1)
		// {
		// 	// Receive in bottom ghost row from the process below
		// 	MPI_Irecv(&E_prev_local[(m_local+1)*(n_local+2)+1], n_local, MPI_DOUBLE, myrank+cb.px, 0, MPI_COMM_WORLD, &req[7]);
		// }
		
		// if(col != cb.px-1) MPI_Wait(&req[0], MPI_STATUS_IGNORE);
		// if(col != 0) MPI_Wait(&req[1], MPI_STATUS_IGNORE);
		// if(row != cb.py-1) MPI_Wait(&req[2], MPI_STATUS_IGNORE);
		// if(row != 0) MPI_Wait(&req[3], MPI_STATUS_IGNORE);
		// if(col != 0) MPI_Wait(&req[4], MPI_STATUS_IGNORE);
		// if(col != cb.px-1) MPI_Wait(&req[5], MPI_STATUS_IGNORE);
		// if(row != 0) MPI_Wait(&req[6], MPI_STATUS_IGNORE);
		// if(row != cb.py-1) MPI_Wait(&req[7], MPI_STATUS_IGNORE);
		// //MPI_Barrier(MPI_COMM_WORLD);
	}
#else
    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }
#endif
//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
#ifdef _MPI_
	// Solve for the excitation, a PDE
	innerBlockRowStartIndex = (n_local+2)+1;
	innerBlockRowEndIndex = (((m_local+2)*(n_local+2) - 1) - (n_local)) - (n_local+2);
	
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n_local+2)) {
        E_tmp = E_local + j;
	E_prev_tmp = E_prev_local + j;
        R_tmp = R_local + j;
	for(i = 0; i < n_local; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n_local+2)]+E_prev_tmp[i-(n_local+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
#else
#ifdef _MPI_
	// Solve for the excitation, a PDE
	innerBlockRowStartIndex = (n_local+2)+1;
	innerBlockRowEndIndex = (((m_local+2)*(n_local+2) - 1) - (n_local)) - (n_local+2);
	
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n_local+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n_local; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n_local+2)]+E_prev_tmp[i-(n_local+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n_local+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n_local; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
#ifdef _MPI_
   swap(E_local, E_prev_local);
#else
   swap(E, E_prev);
#endif

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

#ifdef _MPI_
	stats(E_prev_local, m_local, n_local, &Linf, &sumSq);
	
	if(!cb.noComm)
	{
		double totalLinf=0.0, totalSumSq=0.0;
		MPI_Reduce(&Linf, &totalLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&sumSq, &totalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
		Linf = totalLinf;
		L2 = L2Norm(totalSumSq);
	}
	else
		L2 = L2Norm(sumSq);
#else
  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);
#endif
  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;

#ifdef _MPI_ 
  free(E_local);
  free(E_prev_local);
  free(R_local);
#endif
}

void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
