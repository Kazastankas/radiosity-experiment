#include "radiosity.hpp"
#include "cutil_math.hpp"
#include <SDL/SDL_opengl.h>
#include <vector>

#define SPLIT_UNIT 1.0f
#define MATERIAL_REFLECTIVITY 0.74f
#define GLOBAL_ENERGY 0.0f

namespace radiosity {

bool initialize_scene(Scene* scene) 
{
  // patches
  for (int x = 0; x < (int) (2.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (2.0f / SPLIT_UNIT); y++) {
      Plane outside_light;
      outside_light.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                             ((float) y * SPLIT_UNIT) - 1.0f, 5);
      outside_light.color = make_float3(1, 1, 1);
      outside_light.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      outside_light.y_vec = make_float3(0, SPLIT_UNIT, 0);
      outside_light.x_min = outside_light.y_min = 0;
      outside_light.x_max = outside_light.y_max = 1;
      outside_light.emission = 50;
      outside_light.energy = GLOBAL_ENERGY;
      outside_light.bound = Box(outside_light.corner_pos,
                                outside_light.corner_pos +
                                outside_light.x_vec +
                                outside_light.y_vec,
                                &outside_light);
      scene->patches.push_back(outside_light);
    }
  }
  
  for (int x = 0; x < (int) (10.0f / SPLIT_UNIT); x++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane top_wall;
      top_wall.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f, 5,
                                        ((float) z * SPLIT_UNIT) - 5.0f);
      top_wall.color = make_float3(0, 1, 0);
      top_wall.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      top_wall.y_vec = make_float3(0, 0, SPLIT_UNIT);
      top_wall.x_min = top_wall.y_min = 0;
      top_wall.x_max = top_wall.y_max = 1;
      top_wall.emission = 0;
      top_wall.energy = GLOBAL_ENERGY;
      top_wall.bound = Box(top_wall.corner_pos,
                           top_wall.corner_pos +
                           top_wall.x_vec +
                           top_wall.y_vec,
                           &top_wall);
      
      Plane bot_wall;
      bot_wall.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f, -5,
                                        ((float) z * SPLIT_UNIT) - 5.0f);
      bot_wall.color = make_float3(1, 0, 1);
      bot_wall.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      bot_wall.y_vec = make_float3(0, 0, SPLIT_UNIT);
      bot_wall.x_min = bot_wall.y_min = 0;
      bot_wall.x_max = bot_wall.y_max = 1;
      bot_wall.emission = 0;
      bot_wall.energy = GLOBAL_ENERGY;
      bot_wall.bound = Box(bot_wall.corner_pos,
                           bot_wall.corner_pos +
                           bot_wall.x_vec +
                           bot_wall.y_vec,
                           &bot_wall);
      
      scene->patches.push_back(top_wall);
      scene->patches.push_back(bot_wall);
    }
  }

  
  for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
    for (int z = 0; z < (int) (10.0f / SPLIT_UNIT); z++) {
      Plane left_wall;
      left_wall.corner_pos = make_float3(-5, ((float) y * SPLIT_UNIT) - 5.0f,
                                         ((float) z * SPLIT_UNIT) - 5.0f);
      left_wall.color = make_float3(0, 1, 1);
      left_wall.x_vec = make_float3(0, SPLIT_UNIT, 0);
      left_wall.y_vec = make_float3(0, 0, SPLIT_UNIT);
      left_wall.x_min = left_wall.y_min = 0;
      left_wall.x_max = left_wall.y_max = 1;
      left_wall.emission = 0;
      left_wall.energy = GLOBAL_ENERGY;
      left_wall.bound = Box(left_wall.corner_pos,
                            left_wall.corner_pos +
                            left_wall.x_vec +
                            left_wall.y_vec,
                            &left_wall);
      
      Plane right_wall;
      right_wall.corner_pos = make_float3(5, ((float) y * SPLIT_UNIT) - 5.0f,
                                         ((float) z * SPLIT_UNIT) - 5.0f);
      right_wall.color = make_float3(1, 0, 0);
      right_wall.x_vec = make_float3(0, SPLIT_UNIT, 0);
      right_wall.y_vec = make_float3(0, 0, SPLIT_UNIT);
      right_wall.x_min = right_wall.y_min = 0;
      right_wall.x_max = right_wall.y_max = 1;
      right_wall.emission = 0;
      right_wall.energy = GLOBAL_ENERGY;
      right_wall.bound = Box(right_wall.corner_pos,
                             right_wall.corner_pos +
                             right_wall.x_vec +
                             right_wall.y_vec,
                             &right_wall);
      
      scene->patches.push_back(left_wall);
      scene->patches.push_back(right_wall);
    }
  }
  
  for (int x = 0; x < (int) (10.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane front_wall;
      front_wall.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f,
                                          ((float) y * SPLIT_UNIT) - 5.0f, -5);
      front_wall.color = make_float3(1, 1, 0);
      front_wall.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      front_wall.y_vec = make_float3(0, SPLIT_UNIT, 0);
      front_wall.x_min = front_wall.y_min = 0;
      front_wall.x_max = front_wall.y_max = 1;
      front_wall.emission = 0;
      front_wall.energy = GLOBAL_ENERGY;
      front_wall.bound = Box(front_wall.corner_pos,
                             front_wall.corner_pos +
                             front_wall.x_vec +
                             front_wall.y_vec,
                             &front_wall);
      
      scene->patches.push_back(front_wall);
    }
  }

  for (int x = 0; x < (int) (4.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (10.0f / SPLIT_UNIT); y++) {
      Plane back_wall;
      back_wall.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 5.0f,
                                         ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall.color = make_float3(0, 0, 1);
      back_wall.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall.y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall.x_min = back_wall.y_min = 0;
      back_wall.x_max = back_wall.y_max = 1;
      back_wall.emission = 0;
      back_wall.energy = GLOBAL_ENERGY;
      back_wall.bound = Box(back_wall.corner_pos,
                            back_wall.corner_pos +
                            back_wall.x_vec +
                            back_wall.y_vec,
                            &back_wall);
      
      Plane back_wall_2;
      back_wall_2.corner_pos = make_float3(((float) x * SPLIT_UNIT) + 1.0f,
                                           ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall_2.color = make_float3(0, 0, 1);
      back_wall_2.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_2.y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_2.x_min = back_wall_2.y_min = 0;
      back_wall_2.x_max = back_wall_2.y_max = 1;
      back_wall_2.emission = 0;
      back_wall_2.energy = GLOBAL_ENERGY;
      back_wall_2.bound = Box(back_wall_2.corner_pos,
                              back_wall_2.corner_pos +
                              back_wall_2.x_vec +
                              back_wall_2.y_vec,
                              &back_wall_2);
      
      scene->patches.push_back(back_wall);
      scene->patches.push_back(back_wall_2);
    }
  }

  
  for (int x = 0; x < (int) (2.0f / SPLIT_UNIT); x++) {
    for (int y = 0; y < (int) (4.0f / SPLIT_UNIT); y++) {
      Plane back_wall_3;
      back_wall_3.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                           ((float) y * SPLIT_UNIT) - 5.0f, 5);
      back_wall_3.color = make_float3(0, 0, 1);
      back_wall_3.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_3.y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_3.x_min = back_wall_3.y_min = 0;
      back_wall_3.x_max = back_wall_3.y_max = 1;
      back_wall_3.emission = 0;
      back_wall_3.energy = GLOBAL_ENERGY;
      back_wall_3.bound = Box(back_wall_3.corner_pos,
                              back_wall_3.corner_pos +
                              back_wall_3.x_vec +
                              back_wall_3.y_vec,
                              &back_wall_3);
      
      Plane back_wall_4;
      back_wall_4.corner_pos = make_float3(((float) x * SPLIT_UNIT) - 1.0f,
                                           ((float) y * SPLIT_UNIT) + 1.0f, 5);
      back_wall_4.color = make_float3(0, 0, 1);
      back_wall_4.x_vec = make_float3(SPLIT_UNIT, 0, 0);
      back_wall_4.y_vec = make_float3(0, SPLIT_UNIT, 0);
      back_wall_4.x_min = back_wall_4.y_min = 0;
      back_wall_4.x_max = back_wall_4.y_max = 1;
      back_wall_4.emission = 0;
      back_wall_4.energy = GLOBAL_ENERGY;
      back_wall_4.bound = Box(back_wall_4.corner_pos,
                              back_wall_4.corner_pos +
                              back_wall_4.x_vec +
                              back_wall_4.y_vec,
                              &back_wall_4);
      
      scene->patches.push_back(back_wall_3);
      scene->patches.push_back(back_wall_4);
    }
  }
 
  for (size_t i = 0; i < scene->patches.size(); i++) {
    scene->tree.insert(&scene->patches[i].bound);
  }

  return true;
}


void draw_plane(Plane *p)
{
	float3 point1 = p->corner_pos;
	float3 point2 = p->corner_pos + p->x_vec;
	float3 point3 = p->corner_pos + p->x_vec + p->y_vec;
	float3 point4 = p->corner_pos + p->y_vec;

	glBegin(GL_QUADS);

	glColor3f(p->color.x, p->color.y, p->color.z);
	glVertex3f(point1.x, point1.y, point1.z);
	glVertex3f(point2.x, point2.y, point2.z);
	glVertex3f(point3.x, point3.y, point3.z);
	glVertex3f(point4.x, point4.y, point4.z);

	glEnd();
}

void draw_scene(Scene *s)
{
	std::vector<Plane>::iterator it;
	for(it=s->patches.begin(); it < s->patches.end(); it++)
		draw_plane(&(*it));
}

}

