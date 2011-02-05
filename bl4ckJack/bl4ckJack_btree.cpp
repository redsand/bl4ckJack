
#include "bl4ckJack_btree.h"
#include <Qt>
#include <QDebug>
#include <QString>

/* ******************************************************************* */
/* *******************   NODE CLASS IMPLEMENTATION  ****************** */
/* ******************************************************************* */
Node::Node(void)
{
   left = right = 0;
   data = NULL;
   len = 0;
}
/* ******************************************************************* */
Node::Node(void *el, size_t s)
{
   left = right = 0;
   data = el;
   len = s;
}
/* ******************************************************************* */
/* ****************   BINSTREE CLASS IMPLEMENTATION   **************** */
/* ******************************************************************* */
BinSTree::BinSTree(void)
{
   root = current = 0;
   count = 0;
   error = false;
}

size_t BinSTree::getCount() {
	return this->count;
}
/* ******************************************************************* */
bool BinSTree::find(void *key, size_t s)
{
   Node *p = findNode(key, s);
   if (p == 0)
      error = true;
   else
   {
       error = false;
       current = p;
   }
   return ! error;
}
/* ******************************************************************* */
void BinSTree::insert(void *el, size_t s)
{
	this->count++;
	if (root == 0) {
		root = new Node(el, s); 
		error = false;
	} else {
      Node *p = findNode(el, s);
      if (p == 0) {
		  Node *parent = root;            // assume root is parent
		  if (p != root)                  // can find parent, so let's do
			parent = findParent(el, s);
		  if (greaterThan((unsigned char *)parent->data, parent->len, (unsigned char *)el, s))
			{
			  parent->left = new Node(el, s); 
			  current = parent->left;
			}
		  else
			{
			  parent->right = new Node(el, s);
			  current = parent->right;
			}
		  error = false; 
	  }
		  else                                // duplicate key, not inserted
			error = true;
    }

}
/* ******************************************************************* */
void BinSTree::traverse(int order)
{
   if (order == PRE_ORDER)
      preorder(root, 0);
   else if (order == IN_ORDER)
      inorder(root, 0);
   else if (order == POST_ORDER)
      postorder(root, 0);
   //else
   //   cout << "no such order " << endl;
}
/* ******************************************************************* */
void *BinSTree::retrieve(size_t *len)
{
  if(len) *len = current->len;
  return current->data;
}
/* ******************************************************************* */
int BinSTree::isFull(void)
{ 
  return 0;
}
/* ******************************************************************* */
int BinSTree::isEmpty(void)
{
  return (root == 0);
}
/* ******************************************************************* */
int BinSTree::hasError(void)
{ 
  return error;
}
/* ******************************************************************* */
void BinSTree::remove(void* key, size_t s)
{
  /* NOTE: This is not at all the easiest possible delete routine.
           However, it comes closest to the algorithm we discussed
	   in class without attempting to be too fancy at all.

           The algorithm could easily be improved by "factoring out"
	   its common elements, and even more by combining some of
	   the cases into one. */

  Node *p = findNode(key, s);             
  if (p == 0)                                 // is node is in the tree ?
    error = true;
  else                                        // yes, let's proceed
  {
	this->count--;
    if ((p->right == 0) && (p->left == 0))       // deleting leaf - easy ...
	{
	  if (p != root)    // (can find parent now ...)   
	    {
	      Node *parent = findParent(key, s);
	      if (lessThan((unsigned char *)parent->data, parent->len, (unsigned char *)key, s))
			parent->right = 0;
	      else
			parent->left = 0;
	    }
	  else
	    root = 0;
          delete(p);
	  error = false;
	  current = root;
	}
    else if ((p->right == 0) && (p->left != 0))  // right subtree empty,
                                                   // left subtree not.
	{
	  if (p != root)    // (can find parent now ...)
	    {
	      Node *parent = findParent(key, s);
	      if (lessThan((unsigned char *)parent->data, parent->len, (unsigned char *)key, s))
			parent->right = p->left;
	      else
			parent->left = p->left;
	    }
	  else
	    root = root->left;
	  delete(p);
	  error = false;
	  current = root;
	}
      else if ((p->right != 0) && (p->left == 0))  // left subtree empty,
                                                   // right subtree not.
	{
	  if (p != root)    // (can find parent now ...)
	    {
	      Node *parent = findParent(key, s);
	      if (lessThan((unsigned char *)parent->data, parent->len, (unsigned char *)key, s))
			parent->right = p->right;
	      else
			parent->left = p->right;
	    }
	  else
	    root = p->right;
	  delete(p);
	  error = false;
	  current = root;
	}
      else                                         // left and right 
                                                   // subtrees not empty
	{
	  Node *righty = findRighty(p->left);
	  Node *parent = findParent(righty->data, righty->len);
	  p->data = righty->data;   // swapping data with righty
	  p->len = righty->len;
	  if (parent != p)
	    parent->right = righty->left;
	  else
	    p->left = righty->left;
	  delete(righty);
	  error = false;
	  current = root;
	}
    }
}
/* ******************************************************************* */
void BinSTree::destroy(void)
{
  destroyNode(root);
  this->count = 0;
}

/* ******************************************************************* */

void BinSTree::inorder(Node *p, int level)
{
   if (p != 0)
     {
       inorder(p->left, level+1);
       //cout << "Node " << p->data << " at level " << level << endl;
       inorder(p->right, level+1);
     }
}
/* ******************************************************************* */
void BinSTree::preorder(Node *p, int level)
{
  if (p != 0)
    {
      //cout << "Node " << p->data << " at level " << level << endl;
      preorder(p->left, level+1);
      preorder(p->right, level+1);
    }
}
/* ******************************************************************* */
void BinSTree::postorder(Node *p, int level)
{ 
  if (p != 0)
    {
      postorder(p->left, level+1);
      postorder(p->right, level+1);
      //cout << "Node " << p->data << " at level " << level << endl;
    }
}
/* ******************************************************************* */
void BinSTree::destroyNode(Node *p)
{
  if (p != 0)
    {
      destroyNode(p->left);
      destroyNode(p->right);
	  delete(p->data);
      delete(p);
    }
}
/* ******************************************************************* */
Node* BinSTree::findParent(void *key, size_t s)
{
   Node *p = root, *q = 0;
   while ((p != 0) && (!equals((unsigned char *)p->data, p->len, (unsigned char *)key, s)))
     {
       q = p;
	   if (greaterThan((unsigned char *)p->data, p->len, (unsigned char *)key, s))
		p = p->left;
       else
		p = p->right;
     }
   return q;
}

bool BinSTree::lessThan(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {


	register unsigned int i=0;
	if(!base) return false;
	if(!compare) return false;

	for(i=0; i < baseLen, i < compareLen; i++) {
		if(compare[i] > base[i])
			return false;
		else if(compare[i] == base[i])
			continue;
		else
			return true;
	}

	return true;
}

bool BinSTree::greaterThan(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {

	//char buf[256];
	register unsigned int i=0;
	if(!base) return false;
	if(!compare) return false;
	//qDebug() << "Comparing 2 hashes of size " << baseLen;
	for(i=0; i < baseLen, i < compareLen; i++) {
		//sprintf_s(buf, sizeof(buf)-1, "cmp %02X with %02X", (unsigned char)compare[i] & 0xff, (unsigned char) base[i] & 0xff);
		//qDebug() << buf;
		
		if(compare[i] < base[i])
			return false;
		else if(compare[i] == base[i])
			continue;
		else
			return true;

	}

	return true;
}

bool BinSTree::equals(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {


	register unsigned int i=0;
	if(!base) return false;
	if(!compare) return false;

	for(i=0; i < baseLen, i < compareLen; i++) {
		if(base[i] != compare[i]) 
			return false;
	}

	return true;
}

/* ******************************************************************* */
Node* BinSTree::findNode(void *key, size_t s)
  /* This function returns a pointer to the node containing the key 
     value 'key' if it is in the tree. Otherwise, it returns null */
{
   Node *p = root;
   while ((p != 0) && (!equals((unsigned char *)p->data, p->len, (unsigned char *)key, s))) {
     if (greaterThan((unsigned char *)p->data, p->len, (unsigned char *)key, s))
       p = p->left;
     else
       p = p->right;
   }
   return p;
}
/* ******************************************************************* */
Node* BinSTree::findRighty(Node *p)
{
  Node *righty = p;
  while (righty->right != 0)
    righty = righty->right;
  //cout << "found right-most node to be: " << righty->data << endl;
  return righty;
}
/* ******************************************************************* */