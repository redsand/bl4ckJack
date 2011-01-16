//                                                    March 6, 1998
//                                                 Bert G. Wachsmuth
//   BinSTree class
//   **************
//   This file contains the source code to implement a Binary Search
//   Tree class, as discussed during the semester. Note that the class
//   should be working (except for isFull which always returns false),
//   but many methods could and should be optimized further. That is
//   especially true for 'findNode' and 'findParent' which should be 
//   combined into one method, and 'remove', which should be optimized
//   to be more "esthetically pleasing".
//
#ifndef BINSTREE
#define BINSTREE
#include <iostream>

const int PRE_ORDER = 1;
const int IN_ORDER = 2;
const int POST_ORDER = 3;
const int LEVEL_ORDER = 4;

/* ****************************************************************** */
/* *****************      NODE  CLASS    **************************** */
/* ******************************************************************* */
class Node
{
   public: Node *left;
   public: Node *right;
   public: void *data;
   public: size_t len;

   public: Node(void);
   public: Node(void *, size_t);
};
/* ******************************************************************* */
/* *******************    BINSTREE CLASS    ************************* */
/* ****************************************************************** */
class BinSTree
{     // Fields
   private: Node *root;
   private: Node *current;
   private: int error;

      // Constructor and public methods
   public: BinSTree(void);
   public: void insert(void *, size_t);
   public: void remove(void *, size_t);
   public: void *retrieve(size_t *);
   public: void traverse(int order);
   public: bool find(void *, size_t);
   public: void destroy(void);
   public: int isFull(void);
   public: int isEmpty(void);
   public: int hasError(void);
   public: size_t getCount(void);

   public: bool equals(unsigned char *, size_t, unsigned char *, size_t);
   public: bool lessThan(unsigned char *, size_t, unsigned char *, size_t);
   public: bool greaterThan(unsigned char *, size_t, unsigned char *, size_t);

      // Private (utility) methods
   private: void preorder(Node *, int);
   private: void inorder(Node *, int);
   private: void postorder(Node *, int);
   private: Node* findParent(void *key, size_t);
   private: Node* findNode(void *key, size_t);
   private: Node* findRighty(Node *);
   private: void destroyNode(Node *);

private: size_t count;
};

#endif