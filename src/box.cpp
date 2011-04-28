/**
 * @file box.cpp
 * @brief Box class
 *
 * @author Zizhuang Yang (zizhuang)
 */

#include "box.hpp"
#include "radiosity.hpp"
#include <algorithm>
#include <stdio.h>

namespace radiosity {

Box::Box() : min(make_float3(0.0f)), max(make_float3(0.0f)) { }

Box::Box(const float3& min, const float3& max, struct Plane* parent)
    : min(min), max(max) { this->parent = parent; }

Box::~Box() { }

bool Box::x_mincomp(const Box* b1, const Box* b2)
{
  if (b1->min.x < b2->min.x)
    return true;
  else if (b1->min.x == b2->min.x)
    if (b1->min.y < b2->min.y)
      return true;
    else if (b1->min.y == b2->min.y)
      if (b1->min.z < b2->min.z)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::y_mincomp(const Box* b1, const Box* b2)
{
  if (b1->min.y < b2->min.y)
    return true;
  else if (b1->min.y == b2->min.y)
    if (b1->min.z < b2->min.z)
      return true;
    else if (b1->min.z == b2->min.z)
      if (b1->min.x < b2->min.x)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::z_mincomp(const Box* b1, const Box* b2)
{
  if (b1->min.z < b2->min.z)
    return true;
  else if (b1->min.z == b2->min.z)
    if (b1->min.x < b2->min.x)
      return true;
    else if (b1->min.x == b2->min.x)
      if (b1->min.y < b2->min.y)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::x_maxcomp(const Box* b1, const Box* b2)
{
  if (b1->max.x < b2->max.x)
    return true;
  else if (b1->max.x == b2->max.x)
    if (b1->max.y < b2->max.y)
      return true;
    else if (b1->max.y == b2->max.y)
      if (b1->max.z < b2->max.z)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::y_maxcomp(const Box* b1, const Box* b2)
{
  if (b1->max.y < b2->max.y)
    return true;
  else if (b1->max.y == b2->max.y)
    if (b1->max.z < b2->max.z)
      return true;
    else if (b1->max.z == b2->max.z)
      if (b1->max.x < b2->max.x)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::z_maxcomp(const Box* b1, const Box* b2)
{
  if (b1->max.z < b2->max.z)
    return true;
  else if (b1->max.z == b2->max.z)
    if (b1->max.x < b2->max.x)
      return true;
    else if (b1->max.x == b2->max.x)
      if (b1->max.y < b2->max.y)
        return true;
      else
        return false;
    else
      return false;
  else
    return false;
}

bool Box::x_lchk(const Box* b1, const Box* b2)
{
  return b1->min.x <= b2->max.x;
}

bool Box::y_lchk(const Box* b1, const Box* b2)
{
  return b1->min.y <= b2->max.y;
}

bool Box::z_lchk(const Box* b1, const Box* b2)
{
  return b1->min.z <= b2->max.z;
}

bool Box::x_rchk(const Box* b1, const Box* b2)
{
  return b1->max.x > b2->max.x;
}

bool Box::y_rchk(const Box* b1, const Box* b2)
{
  return b1->max.y > b2->max.y;
}

bool Box::z_rchk(const Box* b1, const Box* b2)
{
  return b1->max.z > b2->max.z;
}

float Box::intersect(float3 e, float3 d, float t0, float t1,
                     struct Plane** out)
{
  // one for each of the 3 dimensions
  float tmin[3];
  float tmax[3];
  float dir[3];
  dir[0] = 1.0f / d.x;
  dir[1] = 1.0f / d.y;
  dir[2] = 1.0f / d.z;
  
  if (dir[0] >= 0.0f) {
    tmin[0] = (min.x - e.x) * dir[0];
    tmax[0] = (max.x - e.x) * dir[0];
  }
  else {
    tmin[0] = (max.x - e.x) * dir[0];
    tmax[0] = (min.x - e.x) * dir[0];
  }
  if (dir[1] >= 0.0f) {
    tmin[1] = (min.y - e.y) * dir[1];
    tmax[1] = (max.y - e.y) * dir[1];
  }
  else {
    tmin[1] = (max.y - e.y) * dir[1];
    tmax[1] = (min.y - e.y) * dir[1];
  }
  
  if (tmin[0] > tmax[1] || tmin[1] > tmax[0])
    return INFTY;
  if (tmin[1] > tmin[0])
    tmin[0] = tmin[1];
  if (tmax[1] < tmax[0])
    tmax[0] = tmax[1];
    
  if (dir[2] >= 0.0f) {
    tmin[2] = (min.z - e.z) * dir[2];
    tmax[2] = (max.z - e.z) * dir[2];
  }
  else {
    tmin[2] = (max.z - e.z) * dir[2];
    tmax[2] = (min.z - e.z) * dir[2];
  }
  
  if (tmin[0] > tmax[2] || tmin[2] > tmax[0])
    return INFTY;
  if (tmin[2] > tmin[0])
    tmin[0] = tmin[2];
  if (tmax[2] < tmax[0])
    tmax[0] = tmax[2];
  
  if (tmin[0] > t1 || tmax[0] < t0)
    return INFTY;
    
  // Return the farthest impact point.
  return std::min(tmax[0], t1);
}

float Box::intersect_parent(float3 e, float3 d, float t0, float t1,
                            struct Plane** out)
{
  float3 p0 = parent->corner_pos;
  float3 p1 = p0 + parent->x_vec;
  float3 p2 = p0 + parent->y_vec;
  
  // column vectors
  float3 c1 = -d;
  float3 c2 = p1 - p0;
  float3 c3 = p2 - p0;
  float det = dot(c1, cross(c2, c3));
    
  // inverted row vectors
  float3 r1 = cross(c2, c3) / det;
  float3 r2 = cross(c3, c1) / det;
  float3 r3 = cross(c1, c2) / det;
    
  // coefficients
  float t_hit = dot(r1, e - p0);
  float u_hit = dot(r2, e - p0);
  float v_hit = dot(r3, e - p0);
   
  if (t_hit >= t0 && t_hit <= t1 &&
      u_hit > parent->x_min + EPSILON && u_hit < parent->x_max - EPSILON &&
      v_hit > parent->y_min + EPSILON && v_hit < parent->y_max - EPSILON)
  {
    if (out) *out = parent;
    return t_hit;
  }
  return t1;
}

}
