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

//! Binary Search Node structure
/**
  * Binary Search Node structure.
  * Binary search node used for holding data and directing during search.
  */
class Node
{
   public: 
		Node *left; /**< left node */
		Node *right; /**<right node */
		void *data; /**< data pointer */
		size_t len; /**< data length */

		Node(void); 
		Node(void *, size_t);
};
/* ******************************************************************* */
/* *******************    BINSTREE CLASS    ************************* */
/* ****************************************************************** */
//! Binary Search Tree Class
/**
 * Binary Search Tree Class
 * Binary Search Tree Class used for creating and searching binary trees. 
 */
class BinSTree {
   
   // Constructor and public methods
   public:
   
		//! Binary Tree Constructor
		/**
		  * Binary Tree Constructor
		  * Creating an object for binary tree management.
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return None
		  */
		  
		BinSTree(void);
		
		//! Binary Tree Insert
		/**
		  * Binary Tree Insert
		  * Insert an object into the binary tree.
		  * @see BinSTree()
		  * @see remove()
		  * @see ~BinSTree()
		  * @return None
		  */
		void insert(void *, size_t);
		
		//! Binary Tree Remove
		/**
		  * Binary Tree Remove
		  * Remove an object from the binary tree.
		  * @see BinSTree()
		  * @see insert()
		  * @see ~BinSTree()
		  * @return None
		  */
		void remove(void *, size_t);
		
		//! Binary Tree Retrieve
		/**
		  * Binary Tree Retrieve
		  * Retrieve an object from the binary tree.
		  * @param pointer to size_t (return length of @return)
		  * @see BinSTree()
		  * @see insert()
		  * @see find()
		  * @see remove()
		  * @see ~BinSTree()
		  * @return void data pointer
		  */
		void *retrieve(size_t *);
		
		//! Binary Tree Traverse
		/**
		  * Binary Tree Traverse
		  * Traverse the binary tree using a specific order.
		  * @see BinSTree()
		  * @see find()
		  * @see ~BinSTree()
		  * @return None
		  */
		  
		void traverse(int order);
		
		//! Binary Tree Find
		/**
		  * Binary Tree Find
		  * Find an object within the binary tree.
		  * @param void pointer to data
		  * @param size_t length of data
		  * @see BinSTree()
		  * @see traverse(0
		  * @see retrieve()
		  * @see ~BinSTree()
		  * @return bool
		  */
		  
		bool find(void *, size_t);
		
		//! Binary Tree Destroy
		/**
		  * Binary Tree Destroy
		  * Destroy all objects within the binary tree.
		  * @see BinSTree()
		  * @see remove()
		  * @see ~BinSTree()
		  * @return None
		  */
		void destroy(void);
		
		//! Binary Tree Is Full
		/**
		  * Binary Tree Is Full
		  * Is BTree full of entries?
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return int
		  */
		int isFull(void);
		
		//! Binary Tree Is Empty
		/**
		  * Binary Tree Is Empty
		  * Is BTree empty of entries?
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return int
		  */
		int isEmpty(void);
		
		//! Binary Tree Has Error
		/**
		  * Binary Tree Has Error
		  * BTree error detection
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return int
		  */
		int hasError(void);
		
		//! Binary Tree Get Count
		/**
		  * Binary Tree Get Count
		  * Get Binary Tree count.
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return size_t
		  */
		size_t getCount(void);
		
		//! Binary Tree Get Root Node
		/**
		  * Binary Tree Get Root Node
		  * Get Binary Tree root node.
		  * @see BinSTree()
		  * @see ~BinSTree()
		  * @return Node *
		  */
		Node *getRootNode(void) {
			   return root;
		}

      // Private (utility) methods
   private:    
		bool equals(unsigned char *, size_t, unsigned char *, size_t);
		bool lessThan(unsigned char *, size_t, unsigned char *, size_t);
		bool greaterThan(unsigned char *, size_t, unsigned char *, size_t);

		void preorder(Node *, int);
		void inorder(Node *, int);
			void postorder(Node *, int);
			Node* findParent(void *key, size_t);
			Node* findNode(void *key, size_t);
			Node* findRighty(Node *);
			void destroyNode(Node *);

   // Fields
   //! Private root node
   Node *root;
   //! Private current node
   Node *current;
   //! Private error tracking
   int error;
	size_t count;
};

#endif