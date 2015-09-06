#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	float valuef = 1245.4545;
	unsigned char value;
	value = (unsigned char) valuef;
	printf("%d\n",value);
	return 0;
}