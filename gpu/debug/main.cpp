#include <iostream>
#include <math.h>
using namespace std;

int main()
{
	int bb = 4;
	int total = bb*(bb+1)/2;
	int lastj=-1;
	for(int k=0;k<total;++k)
	{
		int bi = (int)(((-1+sqrt(1+4*2*k)))/2);//floating point calculation!
		int bj = k-((bi*(bi+1))>>1);
		bi=bb-bi;
		bj=bb-bj+1;
		if (lastj<0)
		{
			lastj=bi;
		}
		else if(lastj!=bi)
		{
			lastj=bi;
			cout << "\n";
		}
		cout << bi << "," << bj << "("<<k<<")   ";
	}
	cout << "\n\n\n";



	#define NC 12

	int d_coords[NC];
	for(int i=0;i<NC;++i) d_coords[i] = i;
	cout << "Coordinates: ";
	for (int i=0;i<NC;++i)
	{
		cout << d_coords[i] << "   ";
	}
	cout << "\n\n\n";

	for(int i=1;i<NC-2;++i)
	{
		cout << "\n";
		//for(int t=0;t<i-1;++t) cout << "      ";
		for(int j=NC-2;j>i;--j)
		{
			cout << i << "," << j << "   ";
		}
	}
	cout << "\n\n\n";

	int maxn = NC-3;
	total = maxn*(maxn+1)/2;
	#define TPB2 6
	unsigned int ncoords = TPB2+1;//number/range of coordinates to copy for each i and j
	
	bb = (maxn+TPB2-1)/TPB2;
	int btotal = bb*(bb+1)/2;
	int cc[2*(TPB2+1)];
	int lastb = -1;
	for(int k=btotal-1;k>-1;--k)
	{
		int bi = (int)(((-1+sqrt(1+4*2*k)))/2);//floating point calculation!
		int bj = k-((bi*(bi+1))>>1);
		//bi=bb-bi-1;
		//bj=bj;
		bi=bb-bi-1;
		int maxbj = bj;
		bj=bb-bj-1-bi;
		if(lastb != bi)
		{
			lastb = bi;
			cout << "\n";
		}
		cout << bi << "," << bj << ","<< maxbj<< "   ";
	}
	cout << "\n\n\n";


	for (int i=0;i<2*(TPB2+1);++i) cc[i]=-1;
	
	for(int k=0;k<btotal;++k)
	{

		cout << "Block " << k << ": ";// << "\n"; 
		
		int bi = (int)(((-1+sqrt(1+4*2*k)))/2);//floating point calculation!
		int bj = k-((bi*(bi+1))>>1);
		//bi=bb-bi-1;
		//bj=bj;
		bi=bb-bi-1;
		int pbj=bj;
		bj=bb-bj-1-bi;
		cout << bi << "," << bj << "\n";
		unsigned int istart = bi*TPB2+1;
		int jstart = (NC-2)-(bj+1)*TPB2+1;//(bb-pbj)*TPB2;//istart+1-pbj*TPB2;////bj*TPB2+2;
		cout << "istart,jstart: " << istart << ", " << jstart << "\n";

		for(int tx=0;tx<TPB2;++tx)
		{
			for(int ty=0;ty<TPB2;++ty)
			{
				unsigned int tid = tx+ty*TPB2;
				//Copy needed coords; only the first 2*(TPB2+1) threads are used.
				if(tid<ncoords)
				{//tx goes from [0, TPB2]
					int id=istart+tid-1;//copy start.
					if(id<NC-2)
					{
						cc[tid]=d_coords[id];
					}
				}
				else if(tid<2*ncoords)
				{//tid goes from [TPB2+1, 2*TPB2+1]
					int id=jstart+(tid-ncoords);
					if(id<NC-bj*TPB2 && id>0)
					{
						cc[tid]=d_coords[id];
					}
				}
				
				int ci = tx+1;
				int cj = ty+ncoords;
				int gi = istart+tx;
				int gj = jstart+ty;
				int blockjbound = NC-1-bj*TPB2;
				if(gi<maxn+1 && gj>gi && gj < blockjbound)
				{
					cout << gi << "," << gj << " (" << tx << "," << ty <<")\n";
					//cout << cc[ci] << "," << cc[cj] << " (" << ci << "," << cj <<")\n";
				}
			}
		}

		cout << "Block Check " << k << ": \n";// << "\n"; 
		for(int tx=0;tx<TPB2;++tx)
		{
			for(int ty=0;ty<TPB2;++ty)
			{
				int ci = tx+1;
				int cj = ty+ncoords;
				int gi = istart+tx;
				int gj = jstart+ty;
				int blockjbound = NC-1-bj*TPB2;
				if(gi<maxn+1 && gj>gi && gj < blockjbound)
				{
					cout << cc[ci] << "," << cc[cj] << " (" << ci << "," << cj <<")\n";
				}
			}
		}

		for (int i=0;i<2*(TPB2+1);++i)
		{
			cout << cc[i] << "   ";
		}
		cout << "\n";
	}



	return 0;
}