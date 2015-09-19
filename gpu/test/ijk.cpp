
#include <iostream>
#include <math.h>
using namespace std;

inline int kfromij(int i,int j)
{
	i-=1;
	j-=2;
	return i+((j*(j+1))>>1);
}
inline void ijfromk(int k,int nc,int *ij)
{
	int i = (int)(((-1+sqrtf(1+4*2*k)))/2);//floating point calculation!
	int j = k-((i*(i+1))>>1);
	//i=nc-i-1;
	//j=nc-j-1-i;
	ij[0] = j+1;
	ij[1] = i+2;
}


main()
{
	int i = 18;
	int j = 34;
	int k = kfromij(i,j);
	cout << "i,j: " << i << "," << j << endl;
	cout << "computed k: " << k << endl;
	int ij[2];
	ijfromk(k,0,ij);
	cout << "computed i,j: " << ij[0] << "," << ij[1] << endl;
	return 0;
}