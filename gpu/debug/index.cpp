#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>



inline int ki(int i)
{
	return i-1;
}
inline int kj(int j)
{
	return j-2;
}
inline int kfromij(int pair[2])
{
	int i=ki(pair[0]);
	int j=kj(pair[1]);
	return i+((j*(j+1))>>1);
}

int main()
{
	int correct[2] = {10,12};
	int k = kfromij(correct);
	int nc = 14;
	
	correct[0] = 124;
	correct[1] = 12476;
	k = kfromij(correct);
	nc = 13509;

	fprintf(stderr,"correct = [%d %d]\n",correct[0],correct[1]);
	fprintf(stderr,"k = %d\n",k);

	//pair[0]=3;
	//pair[1]=3;
	//k = kfromij(pair);
	//nc = 7;
	
	//int kpair[2] = {ki(pair[0]),kj(pair[1])};
	//fprintf(stderr,"original indices = [%d %d]\n",(pair[0]),(pair[1]));
	//fprintf(stderr,"swap indices = [%d %d]\n",kpair[0],kpair[1]);

	
	int swap[2];
	int infer[2];
	swap[0] = (int)(((-1+sqrt(1+4*2*k)))/2);//floating point calculation!
	swap[1] = k-((swap[0]*(swap[0]+1))>>1);
	infer[0] = swap[1]+1;
	infer[1] = swap[0]+2;


	fprintf(stderr,"swap indices = [%d %d]\n",swap[0],swap[1]);
	fprintf(stderr,"infer indices = [%d %d]\n",infer[0],infer[1]);
	
	//pair[0]-=1;
	//pair[1]+=nc-3;
	//pair[0]=pair[1];
	//pair[1] = k-((pair[0]*(pair[0]+1))>>1);

	//fprintf(stderr,"pair[] = [%d %d]\n",pair[0],pair[1]);

	int total = ((nc-2)*(nc-3))/2;
	fprintf(stderr,"total: %d\n",total);
}