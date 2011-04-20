#include <radiosity.hpp>

namespace radiosity {

bool initialize_scene(Scene* scene) 
{
  Light outside_light;
  outside_light.pos = make_float3(0, 0, 10);
  outside_light.color = make_float3(1, 1, 1);
  scene->lights.push_back(outside_light);
  
  Plane top_wall;
  top_wall.corner_pos = make_float3(-5, 5, -5);
  top_wall.color = make_float3(0, 1, 0);
  top_wall.x_vec = make_float3(10, 0, 0);
  top_wall.y_vec = make_float3(0, 0, 10);
  top_wall.x_min = top_wall.y_min = 0;
  top_wall.x_max = top_wall.y_max = 1;
  
  Plane bot_wall;
  bot_wall.corner_pos = make_float3(-5, -5, -5);
  bot_wall.color = make_float3(1, 0, 1);
  bot_wall.x_vec = make_float3(10, 0, 0);
  bot_wall.y_vec = make_float3(0, 0, 10);
  bot_wall.x_min = bot_wall.y_min = 0;
  bot_wall.x_max = bot_wall.y_max = 1;
  
  Plane left_wall;
  left_wall.corner_pos = make_float3(-5, -5, -5);
  left_wall.color = make_float3(0, 1, 1);
  left_wall.x_vec = make_float3(0, 10, 0);
  left_wall.y_vec = make_float3(0, 0, 10);
  left_wall.x_min = left_wall.y_min = 0;
  left_wall.x_max = left_wall.y_max = 1;
  
  Plane right_wall;
  right_wall.corner_pos = make_float3(5, -5, -5);
  right_wall.color = make_float3(1, 0, 0);
  right_wall.x_vec = make_float3(0, 10, 0);
  right_wall.y_vec = make_float3(0, 0, 10);
  right_wall.x_min = right_wall.y_min = 0;
  right_wall.x_max = right_wall.y_max = 1;
  
  Plane front_wall;
  front_wall.corner_pos = make_float3(-5, -5, -5);
  front_wall.color = make_float3(1, 1, 0);
  front_wall.x_vec = make_float3(0, 10, 0);
  front_wall.y_vec = make_float3(10, 0, 0);
  front_wall.x_min = front_wall.y_min = 0;
  front_wall.x_max = front_wall.y_max = 1;
  
  Plane back_wall_1;
  back_wall_1.corner_pos = make_float3(-5, -5, 5);
  back_wall_1.color = make_float3(0, 0, 1);
  back_wall_1.x_vec = make_float3(0, 10, 0);
  back_wall_1.y_vec = make_float3(4, 0, 0);
  back_wall_1.x_min = back_wall_1.y_min = 0;
  back_wall_1.x_max = back_wall_1.y_max = 1;
  
  Plane back_wall_2;
  back_wall_2.corner_pos = make_float3(1, -5, 5);
  back_wall_2.color = make_float3(0, 0, 1);
  back_wall_2.x_vec = make_float3(0, 10, 0);
  back_wall_2.y_vec = make_float3(4, 0, 0);
  back_wall_2.x_min = back_wall_2.y_min = 0;
  back_wall_2.x_max = back_wall_2.y_max = 1;
  
  Plane back_wall_3;
  back_wall_3.corner_pos = make_float3(-1, -5, 5);
  back_wall_3.color = make_float3(0, 0, 1);
  back_wall_3.x_vec = make_float3(0, 4, 0);
  back_wall_3.y_vec = make_float3(2, 0, 0);
  back_wall_3.x_min = back_wall_3.y_min = 0;
  back_wall_3.x_max = back_wall_3.y_max = 1;
   
  Plane back_wall_4;
  back_wall_4.corner_pos = make_float3(-1, 1, 5);
  back_wall_4.color = make_float3(0, 0, 1);
  back_wall_4.x_vec = make_float3(0, 4, 0);
  back_wall_4.y_vec = make_float3(2, 0, 0);
  back_wall_4.x_min = back_wall_4.y_min = 0;
  back_wall_4.x_max = back_wall_4.y_max = 1;
  
  scene->objs.push_back(top_wall);
  scene->objs.push_back(bot_wall);
  scene->objs.push_back(left_wall);
  scene->objs.push_back(right_wall);
  scene->objs.push_back(front_wall);
  scene->objs.push_back(back_wall_1);
  scene->objs.push_back(back_wall_2);
  scene->objs.push_back(back_wall_3);
  scene->objs.push_back(back_wall_4);
  return true;
}

}

