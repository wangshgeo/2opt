#pragma once

#define PI 3.141592
#define RRR 6378.388
#define CDDEG(CD) ((int)(CD))
#define CDMIN(CD) ((CD)-(CDDEG(CD)))
#define CDTOLL(CD) (PI*((CDDEG(CD))+5.0*(CDMIN(CD))/3.0)/180.0)

__device__ inline dtype 
coordinateToLongitudeLatitude(dtype coordinate) 
{
	//convert coordinate input to longitude and latitude in radians.
	int deg = (int) coordinate;
	dtype min = coordinate-deg;
	dtype rad = PI*(deg+5.0*min/3.0)/180.0;
	return rad;
}
__device__ inline dtype 
distanceGEO(dtype x1,dtype y1,dtype x2,dtype y2) 
{
	dtype xlat = coordinateToLongitudeLatitude(x1);
	dtype xlong = coordinateToLongitudeLatitude(y1);
	dtype ylat = coordinateToLongitudeLatitude(x2);
	dtype ylong = coordinateToLongitudeLatitude(y2);
	dtype q1 = cosf( xlong - ylong );
	dtype q2 = cosf( xlat - ylat );
	dtype q3 = cosf( xlat + ylat );
	dtype d12 = (int) ( RRR * acosf( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0);
	return d12;
}

#define DELTA(x,y) ((x)-(y))
#define SQUARE(D1) ((D1)*(D1))
#define EUC2D(x1,y1,x2,y2) (round(sqrt(SQUARE(DELTA(x1,x2))+SQUARE(DELTA(y1,y2)))))
#define EDGESUM(iprevx,iprevy,ix,iy,jx,jy,jnextx,jnexty) (EUC2D((iprevx),(iprevy),(ix),(iy))+EUC2D((jx),(jy),(jnextx),(jnexty)))
#define CHECKSWAP(iprevx,iprevy,ix,iy,jx,jy,jnextx,jnexty) (EDGESUM((iprevx),(iprevy),(ix),(iy),(jx),(jy),(jnextx),(jnexty))-EDGESUM((iprevx),(iprevy),(jx),(jy),(ix),(iy),(jnextx),(jnexty)))

__device__ inline dtype 
distanceEUC_2D(dtype x1,dtype y1,dtype x2,dtype y2) 
{
	dtype dx = x1-x2;
	dtype dy = y1-y2;
	dtype dd = sqrtf(dx*dx+dy*dy);
	return dd;
	//return __float2int_rn(dd);
	//return __fsqrt_rd(dx*dx+dy*dy);
	//return round(dd);
}
__device__ inline int 
distanceEUC_2D_int(dtype x1,dtype y1,dtype x2,dtype y2) 
{
	dtype dx = x1-x2;
	dtype dy = y1-y2;
	//dtype dd = sqrtf(dx*dx+dy*dy);
	//return __float2int_rn(dd);
	return __fsqrt_rn(dx*dx+dy*dy);
	//return round(dd);
}
__device__ inline dtype 
checkSwap_EUC_2D(
	dtype iprevx, dtype iprevy, dtype ix, dtype iy,
	dtype jx, dtype jy, dtype jnextx, dtype jnexty)
{
	dtype dold=distanceEUC_2D(iprevx,iprevy,ix,iy)+
		distanceEUC_2D(jx,jy,jnextx,jnexty);
	dtype dnew=distanceEUC_2D(iprevx,iprevy,jx,jy)+
		distanceEUC_2D(ix,iy,jnextx,jnexty);
	//dtype dold=instance.getDistance(tour[i-1],tour[i])+instance.getDistance(tour[j],tour[j+1]);
	//dtype dnew=instance.getDistance(tour[i-1],tour[j])+instance.getDistance(tour[i],tour[j+1]);
	return dnew-dold;
}