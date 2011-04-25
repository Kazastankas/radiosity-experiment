/**
 * @file kdtree.cpp
 * @brief KD tree class
 *
 * @author Zizhuang Yang (zizhuang)
 */

#include "kdtree.hpp"
#include <algorithm>

namespace radiosity {

KDTree::KDTree()
{
  // Clear list.
  geos.clear();
  
  // Initialize variables.
  bound.min = make_float3(INFTY);
  bound.max = make_float3(-INFTY);
  is_leaf = 1;
  depth = 0;
}

KDTree::~KDTree()
{
  // Clear list.
  geos.clear();

  // Recursively wipe child nodes if not a leaf.
  if (!is_leaf) {
    delete left;
    delete right;
  }
}

void KDTree::draw_bound()
{
  // Just collect all the bounding box data.
  for (size_t i = 0; i < geos.size(); i++) {
    Box *add = geos[i];
    if (add->min.x < bound.min.x)
      bound.min.x = add->min.x;
    if (add->max.x > bound.max.x)
      bound.max.x = add->max.x;
    if (add->min.y < bound.min.y)
      bound.min.y = add->min.y;
    if (add->max.y > bound.max.y)
      bound.max.y = add->max.y;
    if (add->min.z < bound.min.z)
      bound.min.z = add->min.z;
    if (add->max.z > bound.max.z)
      bound.max.z = add->max.z;
  }
}
void KDTree::insert(Box *add)
{
  // Add to the list.
  geos.push_back(add);
}

Box KDTree::get_bounding()
{
  // Return the bounding box.
  return bound;
}

float KDTree::intersect(float3 e, float3 d, float t0, float t1,
                        struct Plane** out)
{
  if (!is_leaf) {
    // Parent branch
    Box l_box, r_box;
    float_t lb_t, rb_t, l_t, r_t;
    
    // Intersect with the bounding boxes of the child nodes.
    l_box = left->get_bounding();
    r_box = right->get_bounding();
    
    // Gets the far-side impact time.
    lb_t = l_box.intersect(e, d, t0, t1, NULL);
    rb_t = r_box.intersect(e, d, t0, t1, NULL);

    if (lb_t == INFTY && rb_t == INFTY) {
      // If no hits, then nothing is hit. (This should never happen)
      return t1;
    }
    else if (lb_t == INFTY) {
      // If left is not hit, try hitting the right.
      r_t = right->intersect(e, d, t0, rb_t + 2.0f * EPSILON, out);
      if (r_t - rb_t < EPSILON) // Force that the impact is in the box
        return r_t;
      else 
        return t1;
    }
    else if (rb_t == INFTY) {
      // If right is not hit, try hitting the left.
      l_t = left->intersect(e, d, t0, lb_t + 2.0f * EPSILON, out);
      if (l_t - lb_t < EPSILON) // Force that the impact is in the box
        return l_t;
      else
        return t1;
    }
    if (rb_t - lb_t > EPSILON) {
      // If left has the smaller far-side intersect time, check it first.
      l_t = left->intersect(e, d, t0, lb_t + 2.0f * EPSILON, out);
      if (l_t - lb_t < EPSILON) // Force that the impact is in the box
        return l_t;
      else {
        // If left is not correctly hit, try hitting the right.
        r_t = right->intersect(e, d, t0, rb_t + 2.0f * EPSILON, out);
        if (r_t - rb_t < EPSILON) // Force that the impact is in the box
          return r_t;
        else
          return t1;
      }
    }
    else {
      // Otherwise, check right first.
      r_t = right->intersect(e, d, t0, rb_t + 2.0f * EPSILON, out);
      if (r_t - rb_t < EPSILON) // Force that the impact is in the box
        return r_t;
      else {
        // If right is not correctly hit, try hitting the left.
        l_t = left->intersect(e, d, t0, lb_t + 2.0f * EPSILON, out);
        if (l_t - lb_t < EPSILON) // Force that the impact is in the box
          return l_t;
        else
          return t1;
      }
    }
  }
  else {
    // Child branch

    float impact = t1;
    float box_hit = t1;
    /* Intersect with all geometries (given by the parent pointer)
     * of each bounding box.
     */
    for (size_t i = 0; i < geos.size(); i++) {
      box_hit = geos[i]->intersect(e, d, t0, impact, NULL);
      if (box_hit - impact < EPSILON)
        impact = geos[i]->intersect_parent(e, d, t0, impact, out);
    }
    
    return impact;
  }
}

void KDTree::split() {
  size_t gs = geos.size();
  
  if (gs <= MAX_KD_ELTS || depth > MAX_KD_DEPTH) {
    // Don't bother if node is small or recursive depth is too high.
    is_leaf = 1;
    return;
  }
  
  // Function pointers.
  static bool (*mincomp)(const Box*, const Box*) = NULL;
  static bool (*maxcomp)(const Box*, const Box*) = NULL;
  static bool (*lchk)(const Box*, const Box*) = NULL;
  static bool (*rchk)(const Box*, const Box*) = NULL;
  
  // Initialize them given our depth - this chooses the axis we use.
  switch (depth % 3) {
    case 0:
      mincomp = &Box::x_mincomp;
      maxcomp = &Box::x_maxcomp;
      lchk = &Box::x_lchk;
      rchk = &Box::x_rchk;
      break;
    case 1:
      mincomp = &Box::y_mincomp;
      maxcomp = &Box::y_maxcomp;
      lchk = &Box::y_lchk;
      rchk = &Box::y_rchk;
      break;
    case 2:
      mincomp = &Box::z_mincomp;
      maxcomp = &Box::z_maxcomp;
      lchk = &Box::z_lchk;
      rchk = &Box::z_rchk;
      break;
    default:
      break;
  }
  
  BoxList list_two;
  list_two.resize(gs);
  copy (geos.begin(), geos.end(), list_two.begin());
  
  // Sort first list by the far end of the bounding box at the right axis.
  std::sort(geos.begin(), geos.end(), maxcomp);
  // Sort second list by the close end of the bounding box at the right axis.
  std::sort(list_two.begin(), list_two.end(), mincomp);
  
  /* NAIVE IMPL
  // Pick the median.
  int opt_idx = gs / 2;
  */
  
  int *splits_l = new int[gs];
  int *splits_r = new int[gs];
  int opt_idx = -1;
  float opt_cost = gs;
  float t_w = bound.max.x - bound.min.x;
  float t_h = bound.max.y - bound.min.y;
  float t_d = bound.max.z - bound.min.z;
  float con_fac = 0.5f / (t_w * t_d + t_w * t_h + t_h * t_d);
  
  for (size_t i = 0; i < gs; i++) {
    splits_l[i] = 0;
    splits_r[i] = 0;
  }
  
  size_t l = 0;
  int r = gs - 1;
  
  for (size_t i = 0; i < gs; i++) {
    while (l < gs && lchk(list_two[l], geos[i])) 
      l++;
    splits_l[i] = l;
    
    while (r > -1 && rchk(geos[r], geos[gs-i-1]))
      r--;
    splits_r[gs-i-1] = gs - r - 1;
  }
  
  for (size_t i = 0; i < gs; i++) {
    float w_1 = t_w;
    float h_1 = t_h;
    float d_1 = t_d;
    float w_2 = t_w;
    float h_2 = t_h;
    float d_2 = t_d;

    // Determine new bounding box according to cutting plane.
    switch (depth % 3) {
      case 0:
        w_1 = geos[i]->max.x - bound.min.x;
        w_2 = t_w - w_1;
        break;
      case 1:
        h_1 = geos[i]->max.y - bound.min.y;
        h_2 = t_h - h_1;
        break;
      case 2:
        d_1 = geos[i]->max.z - bound.min.z;
        d_2 = t_d - d_1;
        break;
      default:
        break;
    }
    
    float s1 = 2.0f * (w_1 * d_1 + w_1 * h_1 + d_1 * h_1);
    float s2 = 2.0f * (w_2 * d_2 + w_2 * h_2 + d_2 * h_2);
    float scost = 0.3f + 1.0f * (s1 * con_fac * splits_l[i] + 
                                  s2 * con_fac * splits_r[i]);
    
    if (scost < opt_cost) {
      opt_cost = scost;
      opt_idx = i;
    }
  }
  
  delete[] splits_l;
  delete[] splits_r;
  list_two.clear();
  
  if (opt_idx == -1) {
    is_leaf = 1;
    return;
  }

  // Attempt to make the child nodes.
  is_leaf = 0;
  left = new KDTree;
  right = new KDTree;

  left->depth = depth + 1;
  right->depth = depth + 1;
  
  left->bound.min = bound.min;
  left->bound.max = bound.max;
  right->bound.min = bound.min;
  right->bound.max = bound.max;

  // Set the bounding boxes of the child nodes according to cutting plane.
  switch (depth % 3) {
    case 0:
      left->bound.max.x = geos[opt_idx]->max.x;
      right->bound.min.x = geos[opt_idx]->max.x;
      break;
    case 1:
      left->bound.max.y = geos[opt_idx]->max.y;
      right->bound.min.y = geos[opt_idx]->max.y;
      break;
    case 2:
      left->bound.max.z = geos[opt_idx]->max.z;
      right->bound.min.z = geos[opt_idx]->max.z;
      break;
    default:
      break;
  }
  
  // Figure out which geometries go into which child (some may go into both)
  for (size_t j = 0; j < gs; j++) {
    if (lchk(geos[j], &left->bound))
      left->insert(geos[j]);
    if (rchk(geos[j], &left->bound))
      right->insert(geos[j]);
  }

  // Clear the parent's list, and try to make nodes from children.
  geos.clear();
  left->split();
  right->split();
}

}

