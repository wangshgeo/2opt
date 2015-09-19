#include <iostream>

int main()
{
	int ij[2] = {2701,11257};
	int diff = ij[1]-ij[0]+1;
	std::cout << (ij[0]+diff>>1) << std::endl;
}
