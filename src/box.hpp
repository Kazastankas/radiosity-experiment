/**
 * @file box.hpp
 * @brief Box class
 *
 * @author Zizhuang Yang (zizhuang)
 */

#ifndef BOX_HPP
#define BOX_HPP

#include "cutil_math.hpp"

namespace radiosity {

// Forward declaration

/**
 * A bounding box.
 */
class Box {
public:  
  // The bounds of the box
  float3 min;
  float3 max;
  
  // Parent pointer
  struct Plane *parent;

  Box();
  Box(const float3& min, const float3& max, struct Plane* parent);
  ~Box();

  static bool x_mincomp(const Box* b1, const Box* b2);
  static bool y_mincomp(const Box* b1, const Box* b2);
  static bool z_mincomp(const Box* b1, const Box* b2);
  static bool x_maxcomp(const Box* b1, const Box* b2);
  static bool y_maxcomp(const Box* b1, const Box* b2);
  static bool z_maxcomp(const Box* b1, const Box* b2);
  static bool x_lchk(const Box* b1, const Box* b2);
  static bool y_lchk(const Box* b1, const Box* b2);
  static bool z_lchk(const Box* b1, const Box* b2);
  static bool x_rchk(const Box* b1, const Box* b2);
  static bool y_rchk(const Box* b1, const Box* b2);
  static bool z_rchk(const Box* b1, const Box* b2);
  
  void initialize() const { };

  float intersect(float3 e, float3 d, float t0, float t1,
                  struct Plane** out);
  float intersect_parent(float3 e, float3 d, float t0, float t1,
                         struct Plane** out);
};

}

#endif

