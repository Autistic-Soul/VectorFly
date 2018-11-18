#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>

inline int fast_readin()
{
	char _c = getchar();
	int _d = 0, _flag = 1;
	while (!isdigit(_c) && _c != '-')
		_c = getchar();
	if (_c == '-')
	{
		_flag = -1;
		_c = getchar();
	}
	while (isdigit(_c))
	{
		_d = _d * 10 + _c - '0';
		_c = getchar();
	}
	return _d * _flag;
}

struct node
{
	int data;
	struct node *next;
} *head, *tau;
int NODE_SIZE = sizeof(node);

int append(int value)
{
	node* _q = (node *)malloc(NODE_SIZE);
}

int m = 0;

int main()
{
	m = fast_readin();
	printf("m == %d\n", m);
	return 0;
}
