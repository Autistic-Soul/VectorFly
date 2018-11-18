#include <cstdio>
    #include <cstdlib>
    #include <iostream>
    using namespace std;
    struct node    {
        int data;
        struct node *next;
    }*head;//节点结构体不再解释
    int m,a,b;
    //任然主函数
    int main()    {
        scanf ("%d",&m);    //读取项数
        head = NULL;
        struct node *p,*q;
        for (int i=1;i<=m;i++)    {
            scanf ("%d",&a);    //读取每一个数
            q = (struct node *)malloc(sizeof(struct node));    //    申请空间
            q->data=a;    //填充数据
            q->next = NULL;
            if (head == NULL)    head = q;
            else            p->next = q;
            p = q;
        }
        scanf ("%d",&b);    //读取待插入数据
        struct node *t; t = head;    //用于便利
        for (t = head;t != NULL;t = t->next)    {
            if (t->next->data >= b || t->next == NULL)    {
                q = (struct node*)malloc(sizeof(struct node));    //申请空间
                q->data=b;    //插入数据
                q->next = t->next;    //右端连接
                t->next = q;    //左端连接
            }
        }
        t = head;
        while (t != NULL)    {
            printf ("%d ",t->data);    //输出
            t = t->next;
        }
        return 0;
}                    //第一次写出链表,开心
