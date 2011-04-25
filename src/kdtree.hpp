/**
 * @file kdtree.hpp
 * @brief Class defnition for a kd tree.
 *
 * @author Zizhuang Yang (zizhuang)
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP

#include "box.hpp"
#include <vector>

#define MAX_KD_ELTS 16
#define MAX_KD_DEPTH 30

namespace radiosity {

/**
 * A kd tree, its bounding box determined by its contents.
 */
class KDTree
{
public:
  typedef std::vector< Box* > BoxList;
  
  // the children of this model
  KDTree* left;
  KDTree* right;

  // The geometries spanned by this tree node.
  BoxList geos;
  
  // Depth, also specifies splitting axis
  int depth;

  // leaf identifier
  int is_leaf;
  
  // bounding box.
  Box bound;

  KDTree();
  ~KDTree();

  void draw_bound();

  void insert(Box *geo);

 	Box get_bounding();

  float intersect(float3 e, float3 d, float t0, float t1,
                  struct Plane** out);
  
  void split();
};

} /* _462 */

#endif /* _462_SCENE_SPHERE_HPP_ */

