#include "scene.hpp"
#include "radiosity.hpp"
#include "cutil_math.hpp"
#include <SDL/SDL_opengl.h>
#include <vector>
#include <stdio.h>

#define MATERIAL_REFLECTIVITY 0.74f
#define GLOBAL_ENERGY 0.0f

#define IS_LIGHT(x) (x < (2.0f / SPLIT_UNIT) * (2.0f / SPLIT_UNIT))

namespace radiosity {

void destroy_scene(Scene* scene)
{
  size_t max_idx = scene->patches.size();

  for (int i = 0; i < max_idx; i++) {
    delete scene->patches[i]->bound;
    delete scene->patches[i];
  }

  scene->patches.clear();
}

bool initialize_scene(Scene* scene) 
{
  // patches
  for (int x = 0; x < (int) (2.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (2.0f / SPLIT_UNIT); y++) {
      Plane* outside_light = new Plane;
      outside_light->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                              ((float) y * SPLIT_UNIT) - 1.0f,
                                              10);
      outside_light->reflectivity = make_float3(1, 1, 1);
      outside_light->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      outside_light->y_vec = make_float3(0, SPLIT_UNIT, 0);
      outside_light->x_min = outside_light->y_min = 0;
      outside_light->x_max = outside_light->y_max = 1;
      outside_light->emission = 30;
      outside_light->energy = GLOBAL_ENERGY;
      outside_light->bound = new Box(outside_light->corner_pos,
                                     outside_light->corner_pos +
                                     outside_light->x_vec +
                                     outside_light->y_vec,
                                     outside_light);
      outside_light->ns[0] = outside_light->ns[1] = outside_light->ns[2] =
      outside_light->ns[3] = outside_light->ns[4] = outside_light->ns[5] =
      outside_light->ns[6] = outside_light->ns[7] = scene->patches.size();
      outside_light->normal_dir = 1;
      scene->patches.push_back(outside_light);
    }
  }

  int side_length = (int) (10.0f / SPLIT_UNIT);
  int wider_length = (int) (6.0f / SPLIT_UNIT);
  int wide_length = (int) (4.0f / SPLIT_UNIT);
  int short_length = (int) (2.0f / SPLIT_UNIT);
  
  for (int x = 0; x < (int) (10.0f / SPLIT_UNIT); x++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane* top_wall = new Plane;
      int cur_index = scene->patches.size();
      top_wall->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f, 5,
                                         ((float) z * SPLIT_UNIT) - 5.0f);
      top_wall->reflectivity = make_float3(0, 1, 0);
      top_wall->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      top_wall->y_vec = make_float3(0, 0, SPLIT_UNIT);
      top_wall->x_min = top_wall->y_min = 0;
      top_wall->x_max = top_wall->y_max = 1;
      top_wall->emission = 0;
      top_wall->energy = GLOBAL_ENERGY;
      top_wall->bound = new Box(top_wall->corner_pos,
                                top_wall->corner_pos +
                                top_wall->x_vec +
                                top_wall->y_vec,
                                top_wall);

      if (z < side_length - 1) {
        if (x > 0) top_wall->ns[0] = cur_index - side_length + 1;
        else top_wall->ns[0] = cur_index;
        top_wall->ns[1] = cur_index + 1;
        if (x < side_length - 1) top_wall->ns[2] = cur_index + side_length + 1;
        else top_wall->ns[2] = cur_index;
      } else {
        top_wall->ns[0] = top_wall->ns[1] = top_wall->ns[2] = cur_index;
      }

      if (x < side_length - 1) top_wall->ns[3] = cur_index + side_length;
      else top_wall->ns[3] = cur_index;

      if (z > 0) {
        if (x < side_length - 1) top_wall->ns[4] = cur_index + side_length - 1;
        else top_wall->ns[4] = cur_index;
        top_wall->ns[5] = cur_index - 1;
        if (x > 0) top_wall->ns[6] = cur_index - side_length - 1;
        else top_wall->ns[6] = cur_index;
      } else {
        top_wall->ns[4] = top_wall->ns[5] = top_wall->ns[6] = cur_index;
      }
      
      if (x > 0) top_wall->ns[7] = cur_index - side_length;
      else top_wall->ns[7] = cur_index;
      
      top_wall->normal_dir = -1;
      
      scene->patches.push_back(top_wall);
    }
  }

  for (int x = 0; x < (int) (10.0f / SPLIT_UNIT); x++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane* bot_wall = new Plane;
      int cur_index = scene->patches.size();
      bot_wall->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f, -5,
                                         ((float) z * SPLIT_UNIT) - 5.0f);
      bot_wall->reflectivity = make_float3(1, 0, 1);
      bot_wall->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      bot_wall->y_vec = make_float3(0, 0, SPLIT_UNIT);
      bot_wall->x_min = bot_wall->y_min = 0;
      bot_wall->x_max = bot_wall->y_max = 1;
      bot_wall->emission = 0;
      bot_wall->energy = GLOBAL_ENERGY;
      bot_wall->bound = new Box(bot_wall->corner_pos,
                                bot_wall->corner_pos +
                                bot_wall->x_vec +
                                bot_wall->y_vec,
                                bot_wall);

      if (z < side_length - 1) {
        if (x > 0) bot_wall->ns[0] = cur_index - side_length + 1;
        else bot_wall->ns[0] = cur_index;
        bot_wall->ns[1] = cur_index + 1;
        if (x < side_length - 1) bot_wall->ns[2] = cur_index + side_length + 1;
        else bot_wall->ns[2] = cur_index;
      } else {
        bot_wall->ns[0] = bot_wall->ns[1] = bot_wall->ns[2] = cur_index;
      }

      if (x < side_length - 1) bot_wall->ns[3] = cur_index + side_length;
      else bot_wall->ns[3] = cur_index;

      if (z > 0) {
        if (x < side_length - 1) bot_wall->ns[4] = cur_index + side_length - 1;
        else bot_wall->ns[4] = cur_index;
        bot_wall->ns[5] = cur_index - 1;
        if (x > 0) bot_wall->ns[6] = cur_index - side_length - 1;
        else bot_wall->ns[6] = cur_index;
      } else {
        bot_wall->ns[4] = bot_wall->ns[5] = bot_wall->ns[6] = cur_index;
      }
      
      if (x > 0) bot_wall->ns[7] = cur_index - side_length;
      else bot_wall->ns[7] = cur_index;
      
      bot_wall->normal_dir = 1;
      
      scene->patches.push_back(bot_wall);
    }
  }
  
  for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane* left_wall = new Plane;
      int cur_index = scene->patches.size();
      left_wall->corner_pos = make_float3(-5, ((float) y * SPLIT_UNIT) - 5.0f,
                                          ((float) z * SPLIT_UNIT) - 5.0f);
      left_wall->reflectivity = make_float3(0, 1, 1);
      left_wall->x_vec = make_float3(0, 0, SPLIT_UNIT);
      left_wall->y_vec = make_float3(0, SPLIT_UNIT, 0);
      left_wall->x_min = left_wall->y_min = 0;
      left_wall->x_max = left_wall->y_max = 1;
      left_wall->emission = 0;
      left_wall->energy = GLOBAL_ENERGY;
      left_wall->bound = new Box(left_wall->corner_pos,
                                 left_wall->corner_pos +
                                 left_wall->x_vec +
                                 left_wall->y_vec,
                                 left_wall);
      
      if (y < side_length - 1) {
        if (z > 0) left_wall->ns[0] = cur_index + side_length - 1;
        else left_wall->ns[0] = cur_index;
        left_wall->ns[1] = cur_index + side_length;
        if (z < side_length - 1) left_wall->ns[2] = cur_index + side_length + 1;
        else left_wall->ns[2] = cur_index;
      } else {
        left_wall->ns[0] = left_wall->ns[1] = left_wall->ns[2] = cur_index;
      }

      if (z < side_length - 1) left_wall->ns[3] = cur_index + 1;
      else left_wall->ns[3] = cur_index;

      if (y > 0) {
        if (z < side_length - 1) left_wall->ns[4] = cur_index - side_length + 1;
        else left_wall->ns[4] = cur_index;
        left_wall->ns[5] = cur_index - side_length;
        if (z > 0) left_wall->ns[6] = cur_index - side_length - 1;
        else left_wall->ns[6] = cur_index;
      } else {
        left_wall->ns[4] = left_wall->ns[5] = left_wall->ns[6] = cur_index;
      }
      
      if (z > 0) left_wall->ns[7] = cur_index - 1;
      else left_wall->ns[7] = cur_index;
      
      left_wall->normal_dir = 1;
      
      scene->patches.push_back(left_wall);
    }
  }
  
  for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane* right_wall = new Plane;
      int cur_index = scene->patches.size();
      right_wall->corner_pos = make_float3(5, ((float) y * SPLIT_UNIT) - 5.0f,
                                           ((float) z * SPLIT_UNIT) - 5.0f);
      right_wall->reflectivity = make_float3(1, 0, 0);
      right_wall->x_vec = make_float3(0, 0, SPLIT_UNIT);
      right_wall->y_vec = make_float3(0, SPLIT_UNIT, 0);
      right_wall->x_min = right_wall->y_min = 0;
      right_wall->x_max = right_wall->y_max = 1;
      right_wall->emission = 0;
      right_wall->energy = GLOBAL_ENERGY;
      right_wall->bound = new Box(right_wall->corner_pos,
                                  right_wall->corner_pos +
                                  right_wall->x_vec +
                                  right_wall->y_vec,
                                  right_wall);
      
      if (y < side_length - 1) {
        if (z > 0) right_wall->ns[0] = cur_index + side_length - 1;
        else right_wall->ns[0] = cur_index;
        right_wall->ns[1] = cur_index + side_length;
        if (z < side_length - 1)
          right_wall->ns[2] = cur_index + side_length + 1;
        else right_wall->ns[2] = cur_index;
      } else {
        right_wall->ns[0] = right_wall->ns[1] = right_wall->ns[2] = cur_index;
      }

      if (z < side_length - 1) right_wall->ns[3] = cur_index + 1;
      else right_wall->ns[3] = cur_index;

      if (y > 0) {
        if (z < side_length - 1)
          right_wall->ns[4] = cur_index - side_length + 1;
        else right_wall->ns[4] = cur_index;
        right_wall->ns[5] = cur_index - side_length;
        if (z > 0) right_wall->ns[6] = cur_index - side_length - 1;
        else right_wall->ns[6] = cur_index;
      } else {
        right_wall->ns[4] = right_wall->ns[5] = right_wall->ns[6] = cur_index;
      }
      
      if (z > 0) right_wall->ns[7] = cur_index - 1;
      else right_wall->ns[7] = cur_index;
      
      right_wall->normal_dir = -1;
      
      scene->patches.push_back(right_wall);
    }
  }
  
  for (int x = 0; x < (int) (10.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane* front_wall = new Plane;
      int cur_index = scene->patches.size();
      front_wall->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f,
                                           ((float) y * SPLIT_UNIT) - 5.0f, -5);
      front_wall->reflectivity = make_float3(1, 1, 0);
      front_wall->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      front_wall->y_vec = make_float3(0, SPLIT_UNIT, 0);
      front_wall->x_min = front_wall->y_min = 0;
      front_wall->x_max = front_wall->y_max = 1;
      front_wall->emission = 0;
      front_wall->energy = GLOBAL_ENERGY;
      front_wall->bound = new Box(front_wall->corner_pos,
                                  front_wall->corner_pos +
                                  front_wall->x_vec +
                                  front_wall->y_vec,
                                  front_wall);
      
      if (y < side_length - 1) {
        if (x > 0) front_wall->ns[0] = cur_index - side_length + 1;
        else front_wall->ns[0] = cur_index;
        front_wall->ns[1] = cur_index + 1;
        if (x < side_length - 1)
          front_wall->ns[2] = cur_index + side_length + 1;
        else front_wall->ns[2] = cur_index;
      } else {
        front_wall->ns[0] = front_wall->ns[1] = front_wall->ns[2] = cur_index;
      }

      if (x < side_length - 1) front_wall->ns[3] = cur_index + side_length;
      else front_wall->ns[3] = cur_index;

      if (y > 0) {
        if (x < side_length - 1)
          front_wall->ns[4] = cur_index + side_length - 1;
        else front_wall->ns[4] = cur_index;
        front_wall->ns[5] = cur_index - 1;
        if (x > 0) front_wall->ns[6] = cur_index - side_length - 1;
        else front_wall->ns[6] = cur_index;
      } else {
        front_wall->ns[4] = front_wall->ns[5] = front_wall->ns[6] = cur_index;
      }
      
      if (x > 0) front_wall->ns[7] = cur_index - side_length;
      else front_wall->ns[7] = cur_index; 
      
      front_wall->normal_dir = -1;
      
      scene->patches.push_back(front_wall);
    }
  }
  
  int bw_idx = scene->patches.size();
  int bw2_idx = bw_idx + wide_length * side_length;
  int bw3_idx = bw2_idx + wide_length * side_length;
  int bw4_idx = bw3_idx + wide_length * short_length;

  for (int x = 0; x < (int) (4.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane* back_wall = new Plane;
      int cur_index = scene->patches.size();
      back_wall->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f,
                                         ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall->reflectivity = make_float3(0, 0, 1);
      back_wall->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall->y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall->x_min = back_wall->y_min = 0;
      back_wall->x_max = back_wall->y_max = 1;
      back_wall->emission = 0;
      back_wall->energy = GLOBAL_ENERGY;
      back_wall->bound = new Box(back_wall->corner_pos,
                                 back_wall->corner_pos +
                                 back_wall->x_vec +
                                 back_wall->y_vec,
                                 back_wall);
      
      if (y < side_length - 1) {
        if (x > 0) back_wall->ns[0] = cur_index - side_length + 1;
        else back_wall->ns[0] = cur_index;
        back_wall->ns[1] = cur_index + 1;
        if (x < wide_length - 1) back_wall->ns[2] = cur_index + side_length + 1;
        else back_wall->ns[2] = cur_index;
      } else {
        back_wall->ns[0] = back_wall->ns[1] = back_wall->ns[2] = cur_index;
      }

      if (x < wide_length - 1) back_wall->ns[3] = cur_index + side_length;
      else back_wall->ns[3] = cur_index;

      if (y > 0) {
        if (x < wide_length - 1) back_wall->ns[4] = cur_index + side_length - 1;
        else back_wall->ns[4] = cur_index;
        back_wall->ns[5] = cur_index - 1;
        if (x > 0) back_wall->ns[6] = cur_index - side_length - 1;
        else back_wall->ns[6] = cur_index;
      } else {
        back_wall->ns[4] = back_wall->ns[5] = back_wall->ns[6] = cur_index;
      }
      
      if (x > 0) back_wall->ns[7] = cur_index - side_length;
      else back_wall->ns[7] = cur_index; 
     
      // Possibly link up right side
      if (x == wide_length - 1) {
        if (y < wide_length) {
          if (y < wide_length - 1) back_wall->ns[2] = bw3_idx + y + 1;
          back_wall->ns[3] = bw3_idx + y;
          if (y > 0) back_wall->ns[4] = bw3_idx + y - 1;
        }
        if (y >= (side_length - wide_length)) {
          if (y < side_length - 1)
            back_wall->ns[2] = bw4_idx + y + 1 - (side_length - wide_length);
          back_wall->ns[3] = bw4_idx + y - (side_length - wide_length);
          if (y > side_length - wide_length)
            back_wall->ns[4] = bw4_idx + y - 1 - (side_length - wide_length);
        }
      }

      back_wall->normal_dir = 1;
      
      scene->patches.push_back(back_wall);
    }
  }
  
  for (int x = 0; x < (int) (4.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane* back_wall_2 = new Plane;
      int cur_index = scene->patches.size();
      back_wall_2->corner_pos = make_float3(((float) x * SPLIT_UNIT) + 1.0f,
                                            ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall_2->reflectivity = make_float3(0, 0, 1);
      back_wall_2->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_2->y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_2->x_min = back_wall_2->y_min = 0;
      back_wall_2->x_max = back_wall_2->y_max = 1;
      back_wall_2->emission = 0;
      back_wall_2->energy = GLOBAL_ENERGY;
      back_wall_2->bound = new Box(back_wall_2->corner_pos,
                                   back_wall_2->corner_pos +
                                   back_wall_2->x_vec +
                                   back_wall_2->y_vec,
                                   back_wall_2);
      
      if (y < side_length - 1) {
        if (x > 0) back_wall_2->ns[0] = cur_index - side_length + 1;
        else back_wall_2->ns[0] = cur_index;
        back_wall_2->ns[1] = cur_index + 1;
        if (x < wide_length - 1)
          back_wall_2->ns[2] = cur_index + side_length + 1;
        else back_wall_2->ns[2] = cur_index;
      } else {
        back_wall_2->ns[0] = back_wall_2->ns[1] =
        back_wall_2->ns[2] = cur_index;
      }

      if (x < wide_length - 1) back_wall_2->ns[3] = cur_index + side_length;
      else back_wall_2->ns[3] = cur_index;

      if (y > 0) {
        if (x < wide_length - 1)
          back_wall_2->ns[4] = cur_index + side_length - 1;
        else back_wall_2->ns[4] = cur_index;
        back_wall_2->ns[5] = cur_index - 1;
        if (x > 0) back_wall_2->ns[6] = cur_index - side_length - 1;
        else back_wall_2->ns[6] = cur_index;
      } else {
        back_wall_2->ns[4] = back_wall_2->ns[5] =
        back_wall_2->ns[6] = cur_index;
      }
      
      if (x > 0) back_wall_2->ns[7] = cur_index - side_length;
      else back_wall_2->ns[7] = cur_index; 
      
      // Possibly link up left side
      if (x == 0) {
        if (y < wide_length) {
          if (y < wide_length - 1) 
            back_wall_2->ns[0] = bw3_idx + y + 1 + 
                                (wide_length * (short_length - 1));
          back_wall_2->ns[7] = bw3_idx + y + (wide_length * (short_length - 1));
          if (y > 0)
            back_wall_2->ns[6] = bw3_idx + y - 1 +
                                (wide_length * (short_length - 1));
        }
        if (y >= side_length - wide_length) {
          if (y < side_length - 1)
            back_wall_2->ns[0] = bw4_idx + y + 1 +
                                (wide_length * (short_length - 1)) -
                                (side_length - wide_length);
          back_wall_2->ns[7] = bw4_idx + y +
                              (wide_length * (short_length - 1)) -
                              (side_length - wide_length);
          if (y > side_length - wide_length)
            back_wall_2->ns[6] = bw4_idx + y - 1 +
                                (wide_length * (short_length - 1)) -
                                (side_length - wide_length);
        }
      }
      
      back_wall_2->normal_dir = 1;
      
      scene->patches.push_back(back_wall_2);
    }
  }
  
  for (int x = 0; x < (int) (2.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (4.0f / SPLIT_UNIT); y++) {
      Plane* back_wall_3 = new Plane;
      int cur_index = scene->patches.size();
      back_wall_3->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                            ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall_3->reflectivity = make_float3(0, 0, 1);
      back_wall_3->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_3->y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_3->x_min = back_wall_3->y_min = 0;
      back_wall_3->x_max = back_wall_3->y_max = 1;
      back_wall_3->emission = 0;
      back_wall_3->energy = GLOBAL_ENERGY;
      back_wall_3->bound = new Box(back_wall_3->corner_pos,
                                   back_wall_3->corner_pos +
                                   back_wall_3->x_vec +
                                   back_wall_3->y_vec,
                                   back_wall_3);
     
      if (y < wide_length - 1) {
        if (x > 0) back_wall_3->ns[0] = cur_index - wide_length + 1;
        else back_wall_3->ns[0] = cur_index;
        back_wall_3->ns[1] = cur_index + 1;
        if (x < short_length - 1)
          back_wall_3->ns[2] = cur_index + wide_length + 1;
        else back_wall_3->ns[2] = cur_index;
      } else {
        back_wall_3->ns[0] = back_wall_3->ns[1] =
        back_wall_3->ns[2] = cur_index;
      }

      if (x < short_length - 1) back_wall_3->ns[3] = cur_index + wide_length;
      else back_wall_3->ns[3] = cur_index;

      if (y > 0) {
        if (x < short_length - 1)
          back_wall_3->ns[4] = cur_index + wide_length - 1;
        else back_wall_3->ns[4] = cur_index;
        back_wall_3->ns[5] = cur_index - 1;
        if (x > 0) back_wall_3->ns[6] = cur_index - wide_length - 1;
        else back_wall_3->ns[6] = cur_index;
      } else {
        back_wall_3->ns[4] = back_wall_3->ns[5] =
        back_wall_3->ns[6] = cur_index;
      }
      
      if (x > 0) back_wall_3->ns[7] = cur_index - wide_length;
      else back_wall_3->ns[7] = cur_index; 
      
      // Link up left side
      if (x == 0) {
         back_wall_3->ns[0] = bw_idx + y + 1 + 
                              (side_length * (wide_length - 1));
         back_wall_3->ns[7] = bw_idx + y + (side_length * (wide_length - 1));
         if (y > 0)
           back_wall_3->ns[6] = bw_idx + y - 1 +
                                (side_length * (wide_length - 1));
      }

      // Link up right side
      if (x == short_length - 1) {
         back_wall_3->ns[2] = bw2_idx + y + 1;
         back_wall_3->ns[3] = bw2_idx + y;
         if (y > 0) back_wall_3->ns[4] = bw2_idx + y - 1;
      }

      back_wall_3->normal_dir = 1;
      
      scene->patches.push_back(back_wall_3);
    }
  }
  
  for (int x = 0; x < (int) (2.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (4.0f / SPLIT_UNIT); y++) {
      Plane* back_wall_4 = new Plane;
      int cur_index = scene->patches.size();
      back_wall_4->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                            ((float) y * SPLIT_UNIT) + 1.0f, 5);
      back_wall_4->reflectivity = make_float3(0, 0, 1);
      back_wall_4->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_4->y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_4->x_min = back_wall_4->y_min = 0;
      back_wall_4->x_max = back_wall_4->y_max = 1;
      back_wall_4->emission = 0;
      back_wall_4->energy = GLOBAL_ENERGY;
      back_wall_4->bound = new Box(back_wall_4->corner_pos,
                                   back_wall_4->corner_pos +
                                   back_wall_4->x_vec +
                                   back_wall_4->y_vec,
                                   back_wall_4);
      
      if (y < wide_length - 1) {
        if (x > 0) back_wall_4->ns[0] = cur_index - wide_length + 1;
        else back_wall_4->ns[0] = cur_index;
        back_wall_4->ns[1] = cur_index + 1;
        if (x < short_length - 1)
          back_wall_4->ns[2] = cur_index + wide_length + 1;
        else back_wall_4->ns[2] = cur_index;
      } else {
        back_wall_4->ns[0] = back_wall_4->ns[1] =
        back_wall_4->ns[2] = cur_index;
      }

      if (x < short_length - 1) back_wall_4->ns[3] = cur_index + wide_length;
      else back_wall_4->ns[3] = cur_index;

      if (y > 0) {
        if (x < short_length - 1)
          back_wall_4->ns[4] = cur_index + wide_length - 1;
        else back_wall_4->ns[4] = cur_index;
        back_wall_4->ns[5] = cur_index - 1;
        if (x > 0) back_wall_4->ns[6] = cur_index - wide_length - 1;
        else back_wall_4->ns[6] = cur_index;
      } else {
        back_wall_4->ns[4] = back_wall_4->ns[5] =
        back_wall_4->ns[6] = cur_index;
      }
      
      if (x > 0) back_wall_4->ns[7] = cur_index - wide_length;
      else back_wall_4->ns[7] = cur_index; 
      
      // Link up left side
      if (x == 0) {
         back_wall_4->ns[0] = bw_idx + y + 1 + (side_length - wide_length) +
                              (side_length * (wide_length - 1));
         back_wall_4->ns[7] = bw_idx + y + (side_length - wide_length) +
                              (side_length * (wide_length - 1));
         if (y > 0)
           back_wall_4->ns[6] = bw_idx + y - 1 + (side_length - wide_length) +
                                (side_length * (wide_length - 1));
      }

      // Link up right side
      if (x == short_length - 1) {
         back_wall_4->ns[2] = bw2_idx + y + 1 + (side_length - wide_length);
         back_wall_4->ns[3] = bw2_idx + y + (side_length - wide_length);
         if (y > 0) back_wall_4->ns[4] = bw2_idx + y - 1 +
                                         (side_length - wide_length);
      }
      
      back_wall_4->normal_dir = 1;
      
      scene->patches.push_back(back_wall_4);
    }
  }
  
  for (int x = 0; x < (int) (4.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane* inner_wall = new Plane;
      int cur_index = scene->patches.size();
      inner_wall->corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f,
                                           ((float) y * SPLIT_UNIT) - 5.0f, -1);
      inner_wall->reflectivity = make_float3(0.3, 0.3, 0.3);
      inner_wall->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      inner_wall->y_vec = make_float3(0, SPLIT_UNIT, 0);
      inner_wall->x_min = inner_wall->y_min = 0;
      inner_wall->x_max = inner_wall->y_max = 1;
      inner_wall->emission = 0;
      inner_wall->energy = GLOBAL_ENERGY;
      inner_wall->bound = new Box(inner_wall->corner_pos,
                                  inner_wall->corner_pos +
                                  inner_wall->x_vec +
                                  inner_wall->y_vec,
                                  inner_wall);
      
      if (y < side_length - 1) {
        if (x > 0) inner_wall->ns[0] = cur_index - side_length + 1;
        else inner_wall->ns[0] = cur_index;
        inner_wall->ns[1] = cur_index + 1;
        if (x < wide_length - 1)
          inner_wall->ns[2] = cur_index + side_length + 1;
        else inner_wall->ns[2] = cur_index;
      } else {
        inner_wall->ns[0] = inner_wall->ns[1] = inner_wall->ns[2] = cur_index;
      }

      if (x < wide_length - 1) inner_wall->ns[3] = cur_index + side_length;
      else inner_wall->ns[3] = cur_index;

      if (y > 0) {
        if (x < wide_length - 1)
          inner_wall->ns[4] = cur_index + side_length - 1;
        else inner_wall->ns[4] = cur_index;
        inner_wall->ns[5] = cur_index - 1;
        if (x > 0) inner_wall->ns[6] = cur_index - side_length - 1;
        else inner_wall->ns[6] = cur_index;
      } else {
        inner_wall->ns[4] = inner_wall->ns[5] = inner_wall->ns[6] = cur_index;
      }
      
      if (x > 0) inner_wall->ns[7] = cur_index - side_length;
      else inner_wall->ns[7] = cur_index; 
      
      inner_wall->normal_dir = -1;
      
      scene->patches.push_back(inner_wall);
    }
  }
  
  for (int x = 0; x < (int) (4.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane* inner_wall_2 = new Plane;
      int cur_index = scene->patches.size();
      inner_wall_2->corner_pos = make_float3(((float) x * SPLIT_UNIT) + 1.0f,
                                             ((float) y * SPLIT_UNIT) - 5.0f,
                                             -1);
      inner_wall_2->reflectivity = make_float3(0.3, 0.3, 0.3);
      inner_wall_2->x_vec = make_float3(SPLIT_UNIT, 0, 0);
      inner_wall_2->y_vec = make_float3(0, SPLIT_UNIT, 0);
      inner_wall_2->x_min = inner_wall_2->y_min = 0;
      inner_wall_2->x_max = inner_wall_2->y_max = 1;
      inner_wall_2->emission = 0;
      inner_wall_2->energy = GLOBAL_ENERGY;
      inner_wall_2->bound = new Box(inner_wall_2->corner_pos,
                                    inner_wall_2->corner_pos +
                                    inner_wall_2->x_vec +
                                    inner_wall_2->y_vec,
                                    inner_wall_2);
      
      if (y < side_length - 1) {
        if (x > 0) inner_wall_2->ns[0] = cur_index - side_length + 1;
        else inner_wall_2->ns[0] = cur_index;
        inner_wall_2->ns[1] = cur_index + 1;
        if (x < wide_length - 1)
          inner_wall_2->ns[2] = cur_index + side_length + 1;
        else inner_wall_2->ns[2] = cur_index;
      } else {
        inner_wall_2->ns[0] = inner_wall_2->ns[1] =
        inner_wall_2->ns[2] = cur_index;
      }

      if (x < wide_length - 1) inner_wall_2->ns[3] = cur_index + side_length;
      else inner_wall_2->ns[3] = cur_index;

      if (y > 0) {
        if (x < wide_length - 1)
          inner_wall_2->ns[4] = cur_index + side_length - 1;
        else inner_wall_2->ns[4] = cur_index;
        inner_wall_2->ns[5] = cur_index - 1;
        if (x > 0) inner_wall_2->ns[6] = cur_index - side_length - 1;
        else inner_wall_2->ns[6] = cur_index;
      } else {
        inner_wall_2->ns[4] = inner_wall_2->ns[5] =
        inner_wall_2->ns[6] = cur_index;
      }
      
      if (x > 0) inner_wall_2->ns[7] = cur_index - side_length;
      else inner_wall_2->ns[7] = cur_index; 
      
      inner_wall_2->normal_dir = -1;
      
      scene->patches.push_back(inner_wall_2);
    }
  }

  for (size_t i = 0; i < scene->patches.size(); i++) {
    if (!IS_LIGHT(i))
      scene->tree.insert(scene->patches[i]->bound);
  }
  scene->tree.split();

  return true;
}


void draw_plane(Plane *p, size_t idx, Scene *s)
{
	float3 point1 = p->corner_pos;
	float3 point2 = p->corner_pos + p->x_vec;
	float3 point3 = p->corner_pos + p->x_vec + p->y_vec;
	float3 point4 = p->corner_pos + p->y_vec;
 
  float3 colornw = s->patches[p->ns[0]]->color;
  float3 colorn = s->patches[p->ns[1]]->color;
  float3 colorne = s->patches[p->ns[2]]->color;
  float3 colore = s->patches[p->ns[3]]->color;
  float3 colorse = s->patches[p->ns[4]]->color;
  float3 colors = s->patches[p->ns[5]]->color;
  float3 colorsw = s->patches[p->ns[6]]->color;
  float3 colorw = s->patches[p->ns[7]]->color;
  float3 color = p->color;

  bool north_exist = (p->ns[1] != idx);
  bool east_exist = (p->ns[3] != idx);
  bool south_exist = (p->ns[5] != idx);
  bool west_exist = (p->ns[7] != idx);

  float3 vnw_color, vne_color, vsw_color, vse_color;
  int vnw, vne, vsw, vse;

  vnw_color = vne_color = vsw_color = vse_color = color;
  vnw = vne = vsw = vse = 1;

  if (north_exist) {
    vne_color += colorn;
    vne++;
    
    vnw_color += colorn; 
    vnw++;

    if (east_exist) {
      vne_color += colorne;
      vne++;
    }
    if (west_exist) {
      vnw_color += colornw;
      vnw++;
    }
  }
  if (south_exist) {
    vsw_color += colors; 
    vsw++;
    
    vse_color += colors;
    vse++;
    
    if (east_exist) {
      vse_color += colorse;
      vse++;
    }
    if (west_exist) {
      vsw_color += colorsw;
      vsw++;
    }
  }

  if (east_exist) {
    vne_color += colore; 
    vne++;
    
    vse_color += colore;
    vse++;
  }
  if (west_exist) {
    vnw_color += colorw; 
    vnw++;
    
    vsw_color += colorw;
    vsw++;
  }

  vnw_color /= (float) vnw;
  vne_color /= (float) vne;
  vsw_color /= (float) vsw;
  vse_color /= (float) vse;
 
  /*
  printf("idx: %d loc: (%f,%f,%f) \n", idx, p->corner_pos.x, p->corner_pos.y, p->corner_pos.z);
  printf("nw: (%f,%f,%f) n: (%f,%f,%f) ne: (%f,%f,%f)\n", colornw.x, colornw.y, colornw.z, colorn.x, colorn.y, colorn.z, colorne.x, colorne.y, colorne.z);
  printf("w: (%f,%f,%f) e: (%f,%f,%f)\n", colorw.x, colorw.y, colorw.z, colore.x, colore.y, colore.z);
  printf("sw: (%f,%f,%f) s: (%f,%f,%f) se: (%f,%f,%f)\n", colorsw.x, colorsw.y, colorsw.z, colors.x, colors.y, colors.z, colorse.x, colorse.y, colorse.z);
  printf("nwf: (%f,%f,%f) nef: (%f,%f,%f)\n", vnw_color.x, vnw_color.y, vnw_color.z, vne_color.x, vne_color.y, vne_color.z);
  printf("swf: (%f,%f,%f) sef: (%f,%f,%f)\n", vsw_color.x, vsw_color.y, vsw_color.z, vse_color.x, vse_color.y, vse_color.z);
  printf("n?: %d, w?: %d, s?: %d, e?: %d\n", north_exist, west_exist, south_exist, east_exist);
  */

  glBegin(GL_QUADS);

	glColor3f(vsw_color.x, vsw_color.y, vsw_color.z);
	glVertex3f(point1.x, point1.y, point1.z);
	glColor3f(vse_color.x, vse_color.y, vse_color.z);
	glVertex3f(point2.x, point2.y, point2.z);
	glColor3f(vne_color.x, vne_color.y, vne_color.z);
	glVertex3f(point3.x, point3.y, point3.z);
	glColor3f(vnw_color.x, vnw_color.y, vnw_color.z);
	glVertex3f(point4.x, point4.y, point4.z);

	glEnd();
}

void draw_scene(Scene *s)
{
  size_t max = s->patches.size();
	for (size_t i = 0; i < max; i++)
		draw_plane(s->patches[i], i, s);
}

}

