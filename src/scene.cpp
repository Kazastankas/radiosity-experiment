#include <radiosity.hpp>

#define MATERIAL_REFLECTIVITY 0.74

namespace radiosity {

bool initialize_scene(Scene* scene) 
{
  for (int x = 0; x < 10; x++) {
    for (int y = 0; y < 10; y++) {
      Plane outside_light;
      outside_light.corner_pos = make_float3(((float) x * 0.1f) - 1.0f,
                                             ((float) y * 0.1f) - 1.0f, 10);
      outside_light.color = make_float3(1, 1, 1);
      outside_light.x_vec = make_float3(0.1f, 0, 0);
      outside_light.y_vec = make_float3(0, 0.1f, 0);
      outside_light.x_min = outside_light.y_min = 0;
      outside_light.x_max = outside_light.y_max = 1;
      outside_light.emission = 1;
      outside_light.reflectance = 0;
      scene->objs.push_back(outside_light);
    }
  }
  
  for (int x = 0; x < 100; x++) {
    for (int z = 0; z < 100; z++) {
      Plane top_wall;
      top_wall.corner_pos = make_float3(((float) x * 0.1f) - 5.0f, 5,
                                        ((float) z * 0.1f) - 5.0f);
      top_wall.color = make_float3(0, 1, 0);
      top_wall.x_vec = make_float3(0.1f, 0, 0);
      top_wall.y_vec = make_float3(0, 0, 0.1f);
      top_wall.x_min = top_wall.y_min = 0;
      top_wall.x_max = top_wall.y_max = 1;
      top_wall.emission = 0;
      top_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      Plane bot_wall;
      bot_wall.corner_pos = make_float3(((float) x * 0.1f) - 5.0f, -5,
                                        ((float) z * 0.1f) - 5.0f);
      bot_wall.color = make_float3(1, 0, 1);
      bot_wall.x_vec = make_float3(0.1f, 0, 0);
      bot_wall.y_vec = make_float3(0, 0, 0.1f);
      bot_wall.x_min = bot_wall.y_min = 0;
      bot_wall.x_max = bot_wall.y_max = 1;
      bot_wall.emission = 0;
      bot_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      scene->objs.push_back(top_wall);
      scene->objs.push_back(bot_wall);
    }
  }

  
  for (int y = 0; y < 100; y++) {
    for (int z = 0; z < 100; z++) {
      Plane left_wall;
      left_wall.corner_pos = make_float3(-5, ((float) y * 0.1f) - 5.0f,
                                         ((float) z * 0.1f) - 5.0f);
      left_wall.color = make_float3(0, 1, 1);
      left_wall.x_vec = make_float3(0, 0.1f, 0);
      left_wall.y_vec = make_float3(0, 0, 0.1f);
      left_wall.x_min = left_wall.y_min = 0;
      left_wall.x_max = left_wall.y_max = 1;
      left_wall.emission = 0;
      left_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      Plane right_wall;
      right_wall.corner_pos = make_float3(5, ((float) y * 0.1f) - 5.0f,
                                         ((float) z * 0.1f) - 5.0f);
      right_wall.color = make_float3(1, 0, 0);
      right_wall.x_vec = make_float3(0, 0.1f, 0);
      right_wall.y_vec = make_float3(0, 0, 0.1f);
      right_wall.x_min = right_wall.y_min = 0;
      right_wall.x_max = right_wall.y_max = 1;
      right_wall.emission = 0;
      right_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      scene->objs.push_back(left_wall);
      scene->objs.push_back(right_wall);
    }
  }
  
  for (int x = 0; x < 100; x++) {
    for (int y = 0; y < 100; y++) {
      Plane front_wall;
      front_wall.corner_pos = make_float3(((float) x * 0.1f) - 5.0f,
                                          ((float) y * 0.1f) - 5.0f, -5);
      front_wall.color = make_float3(1, 1, 0);
      front_wall.x_vec = make_float3(0.1f, 0, 0);
      front_wall.y_vec = make_float3(0, 0.1f, 0);
      front_wall.x_min = front_wall.y_min = 0;
      front_wall.x_max = front_wall.y_max = 1;
      front_wall.emission = 0;
      front_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      scene->objs.push_back(front_wall);
    }
  }

  for (int x = 0; x < 40; x++) {
    for (int y = 0; y < 100; y++) {
      Plane back_wall;
      back_wall.corner_pos = make_float3(((float) x * 0.1f) - 5.0f,
                                         ((float) y * 0.1f) - 5.0f, 5);
      back_wall.color = make_float3(1, 1, 0);
      back_wall.x_vec = make_float3(0.1f, 0, 0);
      back_wall.y_vec = make_float3(0, 0.1f, 0);
      back_wall.x_min = back_wall.y_min = 0;
      back_wall.x_max = back_wall.y_max = 1;
      back_wall.emission = 0;
      back_wall.reflectance = MATERIAL_REFLECTIVITY;
      
      Plane back_wall_2;
      back_wall_2.corner_pos = make_float3(((float) x * 0.1f) + 1.0f,
                                           ((float) y * 0.1f) - 5.0f, 5);
      back_wall_2.color = make_float3(1, 1, 0);
      back_wall_2.x_vec = make_float3(0.1f, 0, 0);
      back_wall_2.y_vec = make_float3(0, 0.1f, 0);
      back_wall_2.x_min = back_wall_2.y_min = 0;
      back_wall_2.x_max = back_wall_2.y_max = 1;
      back_wall_2.emission = 0;
      back_wall_2.reflectance = MATERIAL_REFLECTIVITY;
      
      scene->objs.push_back(back_wall);
      scene->objs.push_back(back_wall_2);
    }
  }

  
  for (int x = 0; x < 20; x++) {
    for (int y = 0; y < 40; y++) {
      Plane back_wall_3;
      back_wall_3.corner_pos = make_float3(((float) x * 0.1f) - 1.0f,
                                           ((float) y * 0.1f) - 5.0f, 5);
      back_wall_3.color = make_float3(1, 1, 0);
      back_wall_3.x_vec = make_float3(0.1f, 0, 0);
      back_wall_3.y_vec = make_float3(0, 0.1f, 0);
      back_wall_3.x_min = back_wall_3.y_min = 0;
      back_wall_3.x_max = back_wall_3.y_max = 1;
      back_wall_3.emission = 0;
      back_wall_3.reflectance = MATERIAL_REFLECTIVITY;
      
      Plane back_wall_4;
      back_wall_4.corner_pos = make_float3(((float) x * 0.1f) - 1.0f,
                                           ((float) y * 0.1f) + 1.0f, 5);
      back_wall_4.color = make_float3(1, 1, 0);
      back_wall_4.x_vec = make_float3(0.1f, 0, 0);
      back_wall_4.y_vec = make_float3(0, 0.1f, 0);
      back_wall_4.x_min = back_wall_4.y_min = 0;
      back_wall_4.x_max = back_wall_4.y_max = 1;
      back_wall_4.emission = 0;
      back_wall_4.reflectance = MATERIAL_REFLECTIVITY;
      
      scene->objs.push_back(back_wall_3);
      scene->objs.push_back(back_wall_4);
    }
  }
  return true;
}

}

